from pathlib import Path
exec(compile(Path('/content/v32_payload/cli_prepare_inputs.py').read_text(),
             'cli_prepare_inputs.py','exec'),{'__name__':'__main__'})
