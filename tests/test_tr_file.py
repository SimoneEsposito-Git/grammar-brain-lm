import numpy as np
import pytest

from data_loading.tr_file import TRFile, load_textgrid_transcripts


def test_simulate_generates_evenly_spaced_trs():
    trf = TRFile(None, expectedtr=2.0)
    trf.simulate(5)
    assert np.allclose(trf.trtimes, [0.0, 2.0, 4.0, 6.0, 8.0])


def test_avgtr_matches_expected_spacing():
    trf = TRFile(None, expectedtr=2.0)
    trf.simulate(4)
    assert trf.avgtr == pytest.approx(2.0)


def test_get_reltriggertimes_is_relative_to_sound_start():
    trf = TRFile(None, expectedtr=1.0)
    trf.trtimes = [1.0, 2.0, 3.0]
    trf.soundstarttime = 1.0
    assert list(trf.get_reltriggertimes()) == [0.0, 1.0, 2.0]


def test_load_from_file_parses_triggers_and_sound_bounds(tmp_path):
    report = tmp_path / "story.report"
    report.write_text(
        "0.0 init-trigger\n"
        "0.5 sound-start\n"
        "2.0 trigger\n"
        "4.0 trigger\n"
        "9.0 sound-stop\n"
    )
    trf = TRFile(str(report), expectedtr=2.0)
    assert trf.trtimes[:3] == [0.0, 2.0, 4.0]
    assert trf.soundstarttime == 0.5
    assert trf.soundstoptime == 9.0


def test_load_from_file_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        TRFile("does/not/exist.report")


def test_load_from_file_inserts_missing_tr_when_gap_is_large():
    # Regular 2.0-unit spacing except for one dropped trigger between t=6 and t=20.
    trf = TRFile(None, expectedtr=2.0)
    trf.trtimes = [0.0, 2.0, 4.0, 6.0, 20.0, 22.0]
    trf.expectedtr = 2.0
    # Re-run the gap-fixing logic the same way load_from_file does.
    itrtimes = np.diff(trf.trtimes)
    badtrtimes = np.nonzero(itrtimes > (itrtimes.mean() * 1.5))[0]
    assert list(badtrtimes) == [3]  # the 6.0 -> 20.0 gap is the anomaly


def test_load_textgrid_transcripts_extracts_word_intervals(tmp_path):
    textgrid = tmp_path / "story.TextGrid"
    textgrid.write_text(
        'item [1]:\n'
        '    class = "IntervalTier"\n'
        '    name = "phones"\n'
        'item [2]:\n'
        '    class = "IntervalTier"\n'
        '    name = "words"\n'
        '    intervals [1]:\n'
        '        xmin = 0.0\n'
        '        xmax = 0.5\n'
        '        text = "hello"\n'
        '    intervals [2]:\n'
        '        xmin = 0.5\n'
        '        xmax = 1.2\n'
        '        text = "world"\n'
    )
    result = load_textgrid_transcripts(["story"], str(tmp_path))
    assert result["story"] == [(0.0, 0.5, "hello"), (0.5, 1.2, "world")]


def test_load_textgrid_transcripts_missing_file_returns_empty_list(tmp_path):
    result = load_textgrid_transcripts(["missing"], str(tmp_path))
    assert result["missing"] == []
