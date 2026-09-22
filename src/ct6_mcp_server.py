# -*- coding: utf-8 -*-
"""simulate_ct6 的 MCP 服务器（stdio / JSON-RPC 2.0）。

把 `ct6_bridge` 的 TCP 桥封装成 MCP 工具，供 Agent 直接控制 CT 仿真程序。

工具集（1 通用 + 4 高频，按用户选定粒度）
----------------------------------------
  ct6_call(op, params)        通用调用：op ∈ {ping,list,get,set,arch,scan,fbp,fermi,shot,view}
  ct6_set_geometry(**params)  写几何/扫描方案参数（alpha, RA, RB, FDD, SFOV_A/B, Z_coverage,
                              rotation_time, sampling_rate, pixel_xy/z, n_ch_set, bowtie_*,
                              pitch, scan_length, slice_thickness, slice_interval, shots_per_source）
  ct6_switch_arch(arch)       切 CT 架构：dual_source|single_wide|dual_layer|pcct|static_multi
  ct6_run_fbp(settle_ms)      执行 FBP 重建并等待落定
  ct6_screenshot()            回传窗口截图（MCP image 内容）

注册方式（harness 的 mcpServers 配置，示例）
-------------------------------------------
  "ct6": {"command": "D:\\\\python\\\\envs\\\\mar\\\\python.exe",
          "args": ["I:\\\\dsw\\\\ct6_mcp_server.py"]}
前提：仿真程序已启动（桥会写发现文件 I:\\dsw\\ct6_bridge.json）。

本文件可独立运行做自检：
  echo {"jsonrpc":"2.0","id":1,"method":"tools/list"} | python ct6_mcp_server.py
"""

import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ct6_bridge as B   # noqa: E402

PROTO = '2024-11-05'
SERVER = {'name': 'ct6-ct-simulator', 'version': '1.0.0'}

OPS = ['ping', 'list', 'get', 'set', 'arch', 'scan', 'fbp', 'fermi', 'shot', 'view', 'tab']

GEOM_KEYS = ['alpha', 'RA', 'RB', 'FDD', 'SFOV_A', 'SFOV_B', 'Z_coverage', 'rotation_time',
             'sampling_rate', 'pixel_xy', 'pixel_z', 'n_ch_set', 'bowtie_sfov', 'bowtie_edge',
             'pitch', 'scan_length', 'slice_thickness', 'slice_interval', 'shots_per_source',
             'nch_auto_sw', 'asymmetric_cb', 'scan_mode']

TOOLS = [
    {'name': 'ct6_call',
     'description': ('通用调用 CT 仿真程序。op: ping(连通) / list(42 个输入变量清单) / '
                     'get(当前输入+121 项派生结果) / set(params,settle_ms) / arch(arch) / '
                     'scan(mode) / fbp(settle_ms) / fermi() / shot() / view()'),
     'inputSchema': {'type': 'object', 'properties': {
         'op': {'type': 'string', 'enum': OPS},
         'params': {'type': 'object', 'description': 'set 的变量字典'},
         'settle_ms': {'type': 'integer', 'description': '写/算后等待防抖重建的毫秒数'}},
         'required': ['op']}},
    {'name': 'ct6_set_geometry',
     'description': ('写系统几何与扫描方案参数（任选若干）：alpha, RA, RB, FDD, SFOV_A, SFOV_B, '
                     'Z_coverage, rotation_time, sampling_rate, pixel_xy, pixel_z, n_ch_set, '
                     'bowtie_sfov, bowtie_edge, pitch, scan_length, slice_thickness, '
                     'slice_interval, shots_per_source, nch_auto_sw, asymmetric_cb, scan_mode'),
     'inputSchema': {'type': 'object', 'properties': {
         k: {'type': ['number', 'boolean', 'string']} for k in GEOM_KEYS}}},
    {'name': 'ct6_switch_arch',
     'description': '切换 CT 架构：dual_source(双源) / single_wide(单源宽体) / dual_layer(双层) / pcct(光子计数) / static_multi(静态24源)',
     'inputSchema': {'type': 'object', 'properties': {
         'arch': {'type': 'string',
                  'enum': ['dual_source', 'single_wide', 'dual_layer', 'pcct', 'static_multi']},
         'settle_ms': {'type': 'integer'}}, 'required': ['arch']}},
    {'name': 'ct6_run_fbp',
     'description': '执行 FBP 重建（LEAP）。返回光谱/静态/锥束等各路径的读数与派生结果。',
     'inputSchema': {'type': 'object', 'properties': {
         'settle_ms': {'type': 'integer', 'default': 8000}}}},
    {'name': 'ct6_screenshot',
     'description': '截取仿真程序窗口，回传 PNG 图像。',
     'inputSchema': {'type': 'object', 'properties': {
         'out_path': {'type': 'string', 'description': '可选：同时保存到该路径'}}}},
]


def _text(s):
    return [{'type': 'text', 'text': s if isinstance(s, str) else
             json.dumps(s, ensure_ascii=False, indent=1)[:60000]}]


def _ok(rid, result):
    return {'jsonrpc': '2.0', 'id': rid, 'result': result}


def _err(rid, code, msg):
    return {'jsonrpc': '2.0', 'id': rid, 'error': {'code': code, 'message': msg}}


def _call_tool(name, args):
    args = args or {}
    try:
        c = B.client()
    except Exception as exc:
        return {'content': _text(f'未连接：仿真程序未运行或发现文件缺失（{exc}）'),
                'isError': True}
    if name == 'ct6_call':
        op = args.get('op')
        kw = dict(args.get('params') or {})
        if 'settle_ms' in args:
            kw['settle_ms'] = args['settle_ms']
        if op not in OPS:
            return {'content': _text(f'未知 op: {op}（可用 {OPS}）'), 'isError': True}
        if op == 'set':
            r = c('set', params=args.get('params') or {},
                  settle_ms=args.get('settle_ms', 2500))
        elif op in ('fbp', 'fermi'):
            r = c(op, settle_ms=args.get('settle_ms', 8000))
        else:
            r = c(op, **kw)
        if op == 'shot' and r.get('png_base64'):
            return {'content': [{'type': 'image', 'data': r['png_base64'],
                                 'mimeType': 'image/png'}]}
        return {'content': _text(r), 'isError': not r.get('ok', False)}
    if name == 'ct6_set_geometry':
        params = {k: v for k, v in args.items() if k in GEOM_KEYS}
        if not params:
            return {'content': _text(f'未提供可写参数（可用 {GEOM_KEYS}）'), 'isError': True}
        r = c('set', params=params, settle_ms=args.get('settle_ms', 2500))
        q = r.get('results', {})
        brief = {k: q.get(k) for k in ('arch_key', 'n_ch', 'n_rows', 'array_total',
                                       'fan_angle_A', 'n_slices', 'rot_total', 'ctdi_rel')
                 if k in q}
        return {'content': _text({'applied': r.get('applied'), 'key_results': brief})}
    if name == 'ct6_switch_arch':
        r = c('arch', arch=args['arch'], settle_ms=args.get('settle_ms', 5000))
        q = r.get('results', {})
        brief = {k: q.get(k) for k in ('arch_key', 'n_src', 'energy_dim', 'rotating',
                                       'views_total', 'views_per_point', 'data_cells',
                                       'g_force_DetA', 't_res_arch_ms') if k in q}
        return {'content': _text(brief)}
    if name == 'ct6_run_fbp':
        r = c('fbp', settle_ms=args.get('settle_ms', 8000))
        q = r.get('results', {})
        brief = {k: q.get(k) for k in ('arch_key', 'leap_rows', 'leap_cols', 'leap_angles',
                                       'sino_shape', 'n_ch', 'n_rows') if k in q}
        return {'content': _text({'leap': brief, 'note': '详细读数见程序 FBP 页签信息栏'})}
    if name == 'ct6_screenshot':
        r = c('shot')
        if not r.get('ok'):
            return {'content': _text(r), 'isError': True}
        out = args.get('out_path')
        if out:
            try:
                with open(out, 'wb') as fh:
                    fh.write(base64.b64decode(r['png_base64']))
            except Exception as exc:
                return {'content': _text(f'保存失败: {exc}'), 'isError': True}
        return {'content': [{'type': 'image', 'data': r['png_base64'],
                             'mimeType': 'image/png'}]}
    return {'content': _text(f'未知工具: {name}'), 'isError': True}


def handle(msg):
    method = msg.get('method')
    rid = msg.get('id')
    if method == 'initialize':
        return _ok(rid, {'protocolVersion': PROTO,
                         'capabilities': {'tools': {}},
                         'serverInfo': SERVER})
    if method in ('notifications/initialized', 'initialized'):
        return None
    if method == 'tools/list':
        return _ok(rid, {'tools': TOOLS})
    if method == 'tools/call':
        p = msg.get('params') or {}
        res = _call_tool(p.get('name'), p.get('arguments'))
        return _ok(rid, res)
    if method == 'ping':
        return _ok(rid, {})
    if rid is None:
        return None
    return _err(rid, -32601, f'未实现的方法: {method}')


def main():
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode('utf-8'))
        except Exception:
            continue
        try:
            resp = handle(msg)
        except Exception as exc:
            resp = _err(msg.get('id'), -32603, f'{type(exc).__name__}: {exc}')
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + '\n')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
