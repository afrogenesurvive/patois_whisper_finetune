.PHONY: all app clean run

all: app

## app — Build the macOS .app bundle via py2app
app:
	./venv/bin/python3 setup_app.py py2app

## run — Run the launcher directly (fast dev loop, no .app build)
run:
	./venv/bin/python3 gui/launcher.py

## clean — Remove build artifacts
clean:
	rm -rf build dist

## deps — Install dependencies needed for the .app build
deps:
	pip3 install pywebview py2app
