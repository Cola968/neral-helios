from pathlib import Path

P = Path(__file__).resolve().parent / "train_guarded_v9.py"


def one(s, a, b, label):
    n = s.count(a)
    if n != 1:
        raise RuntimeError(f"V8.1 thermal anchor {label}: {n}")
    return s.replace(a, b, 1)


def main():
    s = P.read_text(encoding="utf-8")
    if "v8-throttle-temp-c" in s:
        print("Scale V8.1 thermal governor already applied")
        return

    s = one(s,
        "    ap.add_argument('--v7-max-temp-c',type=float,default=82.0)\n    ap.add_argument('--v7-temp-every-updates',type=int,default=50)\n",
        "    ap.add_argument('--v7-max-temp-c',type=float,default=83.0)\n    ap.add_argument('--v7-temp-every-updates',type=int,default=10)\n    ap.add_argument('--v8-throttle-temp-c',type=float,default=78.0)\n    ap.add_argument('--v8-resume-temp-c',type=float,default=75.0)\n",
        "args")

    old = """                temp_c=v7_gpu_temp() if (step%args.v7_temp_every_updates==0) else None
                guard=gpu_guard_reason(reserved_mib=reserved_mib,free_mib=free_mib,temp_c=temp_c,
"""
    new = """                temp_c=v7_gpu_temp() if (step%args.v7_temp_every_updates==0) else None
                if temp_c is not None and temp_c>=args.v8_throttle_temp_c and temp_c<args.v7_max_temp_c:
                    hot=temp_c; started=time.monotonic()
                    print(json.dumps({'event':'v8_thermal_throttle','temp_c':temp_c}),flush=True)
                    while temp_c is not None and temp_c>args.v8_resume_temp_c and time.monotonic()-started<180:
                        time.sleep(5); temp_c=v7_gpu_temp()
                    print(json.dumps({'event':'v8_thermal_resume','temp_c':temp_c,'peak_c':hot,
                                      'cooldown_seconds':round(time.monotonic()-started,1)}),flush=True)
                guard=gpu_guard_reason(reserved_mib=reserved_mib,free_mib=free_mib,temp_c=temp_c,
"""
    s = one(s, old, new, "runtime")
    P.write_text(s, encoding="utf-8")
    print("Scale V8.1 applied: 78C throttle, 75C resume, 83C hard guard")


if __name__ == "__main__":
    main()
