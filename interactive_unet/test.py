from nicegui import ui

ui.input(
    label="Text",
    placeholder="start typing",
    on_change=lambda e: result.set_text("you typed: " + e.value),
    validation={"Input too long": lambda value: len(value) < 20},
)
result = ui.label()

ui.run()
