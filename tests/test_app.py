import pathlib

from streamlit.testing.v1 import AppTest


def test_streamlit_app_smoke():
    app_path = pathlib.Path(__file__).parent.parent / "src" / "dtwin" / "app" / "main.py"
    at = AppTest.from_file(str(app_path.resolve())).run()
    assert not at.exception
    assert "Digital Twin Dashboard" in at.title[0].value
