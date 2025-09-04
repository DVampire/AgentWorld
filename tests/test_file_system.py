import os
import sys
from pathlib import Path
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.filesystem import FileSystemService, FileSystem
from src.filesystem.types import FileReadRequest


async def run():
    base_dir = Path("workdir/fs_demo").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base dir: {base_dir}")

    # Service-level checks
    svc = FileSystemService(base_dir)
    await svc.mkdir(Path("notes"))
    await svc.write_text(Path("notes/hello.txt"), "hello\nworld\n", mode="w")
    read_res = await svc.read(FileReadRequest(path=Path("notes/hello.txt")))
    print("Service read:", read_res.content_text)

    replaced = await svc.replace(Path("notes/hello.txt"), "world", "WORLD")
    print("Service replaced count:", replaced)

    search_name = await svc.search(Path("notes"), query="hello", by="name")
    print("Service search name count:", len(search_name))

    tree_lines = await svc.tree(Path("."))
    print("Service tree (first 3 lines):", " | ".join(tree_lines[:3]))
    
    

    # Adapter-level checks (absolute paths, legacy messages)
    fs = FileSystem(base_dir=base_dir)
    target_abs = os.path.join(base_dir, "notes/hello.txt")
    
    description = await fs.describe()
    print("Adapter description:", description)
    
    read_all = await fs.read_file(str(target_abs))
    print("Adapter read contains:", read_all)

    line_slice = await fs.read_file(str(target_abs), start_line=1, end_line=1)
    print("Adapter slice ok:", line_slice)

    rep_msg = await fs.replace_file_str(str(target_abs), "hello", "HELLO")
    print("Adapter replace msg:", rep_msg)

    copy_abs = base_dir / "notes/hello_copy.txt"
    print(await fs.copy_file(str(target_abs), str(copy_abs)))
    move_abs = base_dir / "notes/moved.txt"
    print(await fs.move_file(str(copy_abs), str(move_abs)))
    rename_abs = base_dir / "notes/renamed.txt"
    print(await fs.rename_file(str(move_abs), str(rename_abs)))

    print(await fs.delete_file(str(rename_abs)))
    print(await fs.delete_directory(str(base_dir / "notes")))


if __name__ == "__main__":
    asyncio.run(run())

