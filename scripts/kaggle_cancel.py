"""Снятие прогона кернела Kaggle (19.09; слово владельца «а ты старую снимай — на будущее»).

В командном клиенте `kaggle kernels` отмены НЕТ (есть только list/push/status/output/logs/delete), поэтому раньше мы
считали, что снять прогон нельзя и просили владельца жать кнопку. На деле в SDK есть
`KernelsApiClient.cancel_kernel_session(ApiCancelKernelSessionRequest(kernel_session_id=...))` -- просто не выведена
в CLI. Этот скрипт её вызывает.

Зачем это нужно: перепуш НЕ снимает предыдущую версию (проверено 17.09: v2 и v3 запустились обе и обе жгли квоту).
Правильный порядок при заторе: снять старую -> запушить новую.

usage:
    .venv/bin/python scripts/kaggle_cancel.py sergueimakarov/arc3-dsv4-duck          # снять текущий прогон
    .venv/bin/python scripts/kaggle_cancel.py sergueimakarov/arc3-dsv4-duck --dry    # только показать, что нашли
"""
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tok = ROOT / ".kaggle/access_token"
if tok.exists():
    os.environ.setdefault("KAGGLE_API_TOKEN", tok.read_text().strip())

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiCancelKernelSessionRequest, ApiGetKernelRequest, ApiGetKernelSessionStatusRequest)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kernel"); ap.add_argument("--dry", action="store_true")
    ap.add_argument("--session-id", type=int, default=None, help="если id сессии известен")
    a = ap.parse_args()
    user, slug = a.kernel.split("/", 1)
    api = KaggleApi(); api.authenticate()

    with api.build_kaggle_client() as k:
        req = ApiGetKernelSessionStatusRequest(); req.user_name = user; req.kernel_slug = slug
        st = k.kernels.kernels_api_client.get_kernel_session_status(req)
        print("статус сейчас: %s" % st.status)
        if str(st.status).endswith("COMPLETE") or "CANCEL" in str(st.status) or "ERROR" in str(st.status):
            print("прогон уже не идёт -- снимать нечего")
            return 0
        sid = a.session_id
        if sid is None:
            gr = ApiGetKernelRequest(); gr.user_name = user; gr.kernel_slug = slug
            r = k.kernels.kernels_api_client.get_kernel(gr)
            meta = getattr(r, "metadata", None)
            sid = int(getattr(meta, "id", 0) or 0)
            print("id кернела: %s (используем как id сессии)" % sid)
        if a.dry:
            print("--dry: ничего не снимаю")
            return 0
        cr = ApiCancelKernelSessionRequest(); cr.kernel_session_id = sid
        resp = k.kernels.kernels_api_client.cancel_kernel_session(cr)
        print("ответ на отмену: %r" % (resp,))
        req2 = ApiGetKernelSessionStatusRequest(); req2.user_name = user; req2.kernel_slug = slug
        st2 = k.kernels.kernels_api_client.get_kernel_session_status(req2)
        print("статус после: %s" % st2.status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
