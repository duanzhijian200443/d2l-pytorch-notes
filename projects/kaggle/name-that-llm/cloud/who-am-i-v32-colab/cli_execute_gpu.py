from pathlib import Path
exec(compile(Path('/content/v32_payload/colab_runner.py').read_text(),
             'colab_runner.py','exec'),{'__name__':'__main__','GPU_MODE':True,'PREPARED_BY_CLI':True})
