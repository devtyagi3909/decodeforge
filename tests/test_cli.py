from decodeforge.cli import main


def test_cli_inspect_pattern(capsys):
    ret = main(["inspect", "--pattern", "foo|bar", "--test", "foo"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Pattern: foo|bar" in captured.out
    assert "ACCEPTED" in captured.out


def test_cli_inspect_rejected(capsys):
    ret = main(["inspect", "--pattern", "foo|bar", "--test", "baz"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "REJECTED" in captured.out
