import pytest

pytest.importorskip("bs4")

from bs4 import BeautifulSoup

from scripts.collect_public_corpus import series_poems, verified_pd
from scripts.collect_50k import verse_candidate


def html(fragment):
    return BeautifulSoup(fragment, "html.parser")


def test_license_banner_with_leading_empty_paragraph():
    notice = "<div class='licenseContainer'><p></p><p>这部作品在1931年1月1日以前出版，在美国属于公有领域。</p></div>"
    assert verified_pd(html(notice))
    restricted = (
        "<div class='licenseContainer'><p>在两岸四地属于公有领域。然而，在美国仍受保护。</p></div>"
    )
    assert not verified_pd(html(restricted))
    assert not verified_pd(html("<p>未注明许可。</p>"))


def test_numbered_series_preserves_linebreaks_and_omits_annotations():
    content = html(
        "<div><p>1<br/>第一行<br/>第二行<sup class='reference'>1</sup><br/><br/>"
        "2、<br/>别的诗<br/>最后一行</p></div>"
    ).div
    assert series_poems("小詩十二首", content) == ["第一行\n第二行", "别的诗\n最后一行"]


def test_verse_screen_blocks_dialogue_catalogue_and_unbroken_prose():
    assert verse_candidate("庭前叶已落，\n山上月初明。")
    assert not verse_candidate("（正末唱）昨夜落秋声。\n（外云）你从哪里来？")
    assert not verse_candidate("詩題：秋日。\n首句：山深人迹少。")
    assert not verse_candidate("吃过午饭以后他一直沿着城墙走，后来遇到了几个以前读书时的朋友。")
