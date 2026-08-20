# Robot-development setup tools

These utilities support the [bilingual environment module](../../docs/setup/README.md).
They are deliberately non-destructive.

| Tool | Purpose | Mutation boundary |
|---|---|---|
| `stack_resolver.py` | Select a reviewed host, ROS 2, and Gazebo teaching profile. | Reads one JSON file and prints output only. |
| `check_env.sh` | Inventory common developer commands and Python packages. | Reads versions and availability only. |
| `stack_matrix.json` | Machine-readable compatibility decisions with primary-source links. | Data only; reviewed on the date recorded in the file. |

```bash
python tools/robotdev/stack_resolver.py --host ubuntu --ubuntu 24.04
bash tools/robotdev/check_env.sh
```

Neither command installs drivers, changes repositories, edits shell startup files, controls a robot, or claims that an optional simulator is supported on an unlisted platform.
