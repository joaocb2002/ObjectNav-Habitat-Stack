def main():
    # Ensure the legacy package is reachable
    from objectnav.legacy import mylib  # noqa: F401

    # Import representative modules you rely on
    from objectnav.legacy.mylib import simsettings  # noqa: F401
    from objectnav.legacy.mylib import simtools     # noqa: F401
    from objectnav.legacy.mylib import probtools    # noqa: F401
    from objectnav.legacy.mylib import dqn_v2       # noqa: F401

    print("Legacy imports OK")

if __name__ == "__main__":
    main()
