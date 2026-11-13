from ald.data.corpus import normalize, prepare, read_source
from ald.utils.files import read_jsonl


def test_stanzas_and_split_grouping(tmp_path):
    assert normalize("  e\u0301  \r\n\r\n  第二  行\r\n") == "é\n\n第二 行"
    raw = tmp_path / "poems.jsonl"
    raw.write_text(
        '{"work_id":"one","text":"窗边只有一只杯。"}\n'
        '{"work_id":"one","text":"坐在灯下读一封旧信。"}\n'
        '{"work_id":"two","text":"无人替船系好绳子。"}\n'
        '{"work_id":"three","text":"风吹动空房间的门。"}\n'
        '{"work_id":"different-id","text":"窗边只有一只杯。"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    manifest = prepare([(raw, "poem")], output)
    assert manifest["removed"]["exact"] == 1
    membership = {}
    for split in ("train", "val", "test"):
        for row in read_jsonl(output / f"{split}.jsonl"):
            assert membership.setdefault(row["work_id"], split) == split
    assert set(membership) == {"one", "two", "three"}


def test_txt_stanzas_and_near_deduplication(tmp_path):
    path = tmp_path / "poems.txt"
    path.write_text(
        "旧灯下的车站很静。\n\n风从月台走过，吹响了一排空椅。\n===\n"
        "旧灯下的车站很静，风从月台走过，吹响了一排空椅！\n===\n"
        "这张地图上没有回家的路。\n",
        encoding="utf-8",
    )
    rows = list(read_source(path, "poem"))
    assert len(rows) == 3
    assert "\n\n" in rows[0]["text"]
    manifest = prepare([(path, "poem")], tmp_path / "data")
    assert manifest["removed"]["near"] == 1
