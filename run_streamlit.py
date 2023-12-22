# Script to run the Streamlit application to allow debugging
from streamlit.web import bootstrap

script = "rpg_book_bot.py"
bootstrap.run(script,
              f"run {script}",
              args=[],
              flag_options={})