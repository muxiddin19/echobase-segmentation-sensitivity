from echobase.gate import passed


def test_pilot_g1_text_does_not_open_the_gate():
    assert not passed("RUN\n- **Phase:** confirmatory runs. G1 freeze done 2026-10-27\n## Log\n")
    assert not passed("RUN\n- **EchoBase G1:** soon\n## Log\n")
    assert not passed("RUN\n## Log\n- **EchoBase G1:** 2026-11-03\n")  # only the header counts


def test_marker_opens_the_gate():
    assert passed("RUN\n- **EchoBase G1:** 2026-11-03\n## Log\n- old entries\n")

