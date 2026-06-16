import ast
import inspect
import textwrap
import types
import torch
import time

class ExitToMainException(Exception):
    """自定义异常，用于直接跳出到main函数"""
    def __init__(self, message="", exit_code=0, data=None):
        self.message = message
        self.exit_code = exit_code
        self.data = data
        super().__init__(message)

def _get_assigned_names(node):
    names = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    else:
        targets = []

    def collect(target):
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect(elt)
        # 忽略属性赋值 self.x 等
    for t in targets:
        collect(t)
    return names

class InjectCaptureTransformer(ast.NodeTransformer):
    def __init__(self, target_names, injected_name, mode="first"):
        super().__init__()
        self.target_names = set(target_names)
        self.injected_name = injected_name
        self.mode = mode  # "first" or "last"

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # === 第一步：扫描所有赋值语句，按变量收集其出现的语句 ===
        assignments = {name: [] for name in self.target_names}

        for stmt in node.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                assigned_names = _get_assigned_names(stmt)
                for name in assigned_names:
                    if name in self.target_names:
                        assignments[name].append(stmt)

        # === 第二步：确定每个变量在哪个 stmt 后插入 capture ===
        # 我们现在要记录：在哪个 stmt 后插入？插入时捕获哪些 name？
        insert_info = {}  # stmt_node -> set of names to capture after this stmt

        for name in self.target_names:
            stmts = assignments[name]
            if not stmts:
                continue
            target_stmt = stmts[0] if self.mode == "first" else stmts[-1]
            if target_stmt not in insert_info:
                insert_info[target_stmt] = set()
            insert_info[target_stmt].add(name)

        # === 第三步：构建新 body，在指定 stmt 后插入 capture 调用 ===
        new_body = []
        for stmt in node.body:
            new_body.append(stmt)
            if stmt in insert_info:
                # 获取在此语句后需要 capture 的所有变量名
                names_to_capture = sorted(insert_info[stmt])  # 排序确保确定性
                for name in names_to_capture:
                    call = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id=self.injected_name, ctx=ast.Load()),
                            args=[
                                ast.Constant(value=name),
                                ast.Name(id=name, ctx=ast.Load())
                            ],
                            keywords=[]
                        )
                    )
                    new_body.append(call)

        node.body = new_body
        return node




def instrument_forward_and_capture(module, var_names, mode="last", capture_until_num=None):
    if isinstance(var_names, str):
        var_names = [var_names]
    var_names = list(var_names)
    original_forward = module.forward
    features = {}
    features['_target_names'] = var_names  # 传递给 capture 函数用作计数初始化

    def _make_capture(features_dict, mode_flag, capture_until_num):
        target_names = features_dict.get('_target_names', [])
        capture_count = {name: 0 for name in target_names}

        def _capture(name, value):
            try:
                is_tensor = isinstance(value, torch.Tensor)
            except Exception:
                is_tensor = False
            saved = value.detach().clone() if is_tensor else value

            # 更新最新值
            try:
                features_dict[name] = torch.cat([features_dict[name],saved], dim=0)
            except KeyError:
                features_dict[name] = saved
            except TypeError:
                features_dict[name] = saved


            if 'start_time' not in features_dict:
                features_dict['start_time'] = time.time()

            # 计数 +1
            if name not in capture_count:
                capture_count[name] = 0
            capture_count[name] += 1
            # print(f"-------------------------------------capture_count[{name}]:{capture_count[name]}-------------------------------------")
            # 判断是否达到周期阈值
            if capture_until_num and (capture_count[name] % capture_until_num == 0):
                raise ExitToMainException(
                    message=f"[Capture#{capture_count[name]}] '{name}' captured {capture_until_num}x cycle → Break!",
                    exit_code=0,
                    data=saved
                )

            return value

        # === 关键：把计数器暴露为属性，供外部操作 ===
        _capture.__dict__['capture_count'] = capture_count
        _capture.__dict__['reset_counter'] = lambda names=None: _reset_count(names)
        
        def _reset_count(names=None):
            """重置计数器
            Args:
                names: str or list of str, 要重置的变量名；None 表示全部
            """
            if names is None:
                vars_to_reset = target_names
            elif isinstance(names, str):
                vars_to_reset = [names]
            else:
                vars_to_reset = names
            
            for name in vars_to_reset:
                if name in capture_count:
                    capture_count[name] = 0

        return _capture

    capture_func = _make_capture(features, mode, capture_until_num)

    try:
        src = inspect.getsource(original_forward)
    except Exception as e:
        raise RuntimeError(f"Cannot get source: {e}")

    src = textwrap.dedent(src)
    try:
        mod_ast = ast.parse(src)
    except SyntaxError as e:
        raise RuntimeError(f"Failed to parse source: {e}")

    g = original_forward.__globals__
    base_injected = "_capture_injected"
    injected_name = base_injected
    if injected_name in g:
        i = 1
        while f"{base_injected}_{i}" in g:
            i += 1
        injected_name = f"{base_injected}_{i}"

    # 关键：把 mode 传进去
    transformer = InjectCaptureTransformer(var_names, injected_name, mode=mode)
    mod_ast = transformer.visit(mod_ast)
    ast.fix_missing_locations(mod_ast)

    g[injected_name] = capture_func
    prev_global_forward = g.get(original_forward.__name__, None)

    filename = inspect.getsourcefile(original_forward) or "<string>"
    compiled = compile(mod_ast, filename, "exec")
    exec(compiled, g)

    new_forward = g.get(original_forward.__name__)
    if new_forward is None or not isinstance(new_forward, types.FunctionType):
        if injected_name in g and g[injected_name] is capture_func:
            del g[injected_name]
        raise RuntimeError("Failed to create new forward function.")

    bound_new_forward = new_forward.__get__(module, module.__class__)
    module.forward = bound_new_forward

    if prev_global_forward is None:
        if g.get(original_forward.__name__) is new_forward:
            del g[original_forward.__name__]
    else:
        g[original_forward.__name__] = prev_global_forward

    def restore():
        module.forward = original_forward
        if injected_name in g and g[injected_name] is capture_func:
            del g[injected_name]

    # 暴露 reset_counter
    def reset_counter(names=None):
        """重置 capture 计数器"""
        if hasattr(capture_func, 'reset_counter'):
            capture_func.reset_counter(names)

    return features, restore, reset_counter

########################################
class C(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(10, 10)

    def forward(self, x):
        h = self.proj(x)
        tmp = torch.relu(h)   # 我们想捕获 tmp（没有返回）
        print("first tmp", tmp)
        tmp = tmp * 2         # 再次修改 tmp
        # print("second tmp", tmp)
        tmp = tmp -10
        print("last tmp", tmp)
        out = tmp + 1
        return out

class B(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.C = C()
    def forward(self, x):
        return self.C(x)

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.B = B()
    def forward(self, x):
        return self.B(x)

if __name__=="__main__":
    # ====== 使用插桩器 ======
    model = A()
    x = torch.randn(2, 10)

    # --- 捕获第一次赋值 ---
    features_first, restore_first = instrument_forward_and_capture(
        model.B.C, ["tmp"], mode="first", capture_until_num=3
    )
    out = model(x)
    print("111 Captured mode=First:", features_first["tmp"])

    x = torch.randn(2, 10)
    out = model(x)
    print("222 Captured mode=First:", features_first["tmp"])

    x = torch.randn(2, 10)
    out = model(x)
    print("333 Captured mode=First:", features_first["tmp"])

    x = torch.randn(2, 10)
    out = model(x)
    print("444 Captured mode=First:", features_first["tmp"])

    x = torch.randn(2, 10)
    out = model(x)
    print("555 Captured mode=First:", features_first["tmp"])

    restore_first()


    # --- 捕获最后一次赋值 ---
    features_last, restore_last = instrument_forward_and_capture(
        model.B.C, ["tmp"], mode="last"
    )
    out = model(x)
    print("Captured mode=Last:", features_last["tmp"])
    restore_last()
