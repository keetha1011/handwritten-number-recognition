import flet as ft
import numpy as np
from PIL import Image, ImageDraw
import io
from Prediction import predict


def main(page: ft.Page):
    page.title = "Handwritten Digit Recognition"
    page.window_width = 370
    page.window_height = 500

    class DrawingCanvas(ft.UserControl):
        def __init__(self, width, height):
            super().__init__()
            self.width = width
            self.height = height
            self.lines = []
            self.drawing = False

        def build(self):
            return ft.GestureDetector(
                on_pan_start=self.on_pan_start,
                on_pan_update=self.on_pan_update,
                on_pan_end=self.on_pan_end,
                content=ft.Container(
                    width=self.width,
                    height=self.height,
                    bgcolor=ft.colors.BLACK,
                    border_radius=8,
                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                    content=ft.Stack(controls=self.lines)
                )
            )

        def on_pan_start(self, e: ft.DragStartEvent):
            self.drawing = True
            self.add_line(e.local_x, e.local_y)

        def on_pan_update(self, e: ft.DragUpdateEvent):
            if self.drawing:
                self.add_line(e.local_x, e.local_y)

        def on_pan_end(self, e: ft.DragEndEvent):
            self.drawing = False

        def add_line(self, x, y):
            line = ft.Container(
                bgcolor=ft.colors.WHITE,
                width=20,
                height=20,
                left=x - 5,
                top=y - 5,
                border_radius=10
            )
            self.lines.append(line)
            self.update()

        def clear(self):
            self.lines.clear()
            self.update()

        def get_image(self):
            img = Image.new("RGB", (self.width, self.height), color="black")
            draw = ImageDraw.Draw(img)
            for line in self.lines:
                draw.ellipse([line.left, line.top, line.left + 10, line.top + 10], fill="white")
            return img

    canvas = DrawingCanvas(350, 350)

    result_text = ft.Text(size=20)

    def clear_canvas():
        canvas.clear()
        result_text.value = ""
        page.update()

    def predict_digit():
        img = canvas.get_image()
        img = img.resize((28, 28)).convert('L')

        x = np.asarray(img)
        vec = x.reshape(1, 784)

        # Load Thetas (make sure these files are in the correct location)
        Theta1 = np.loadtxt('Theta1.txt')
        Theta2 = np.loadtxt('Theta2.txt')

        # Predict
        pred = predict(Theta1, Theta2, vec / 255)

        # Update result
        result_text.value = f"Digit = {pred[0]}"
        page.update()

    # UI Layout
    page.add(
        ft.Column([
            ft.Text("Handwritten Digit Recognition", size=25, color=ft.colors.BLUE),
            canvas,
            ft.Row([
                ft.ElevatedButton("Clear Canvas", on_click=lambda _: clear_canvas()),
                ft.ElevatedButton("Predict", on_click=lambda _: predict_digit()),
            ], alignment=ft.MainAxisAlignment.CENTER),
            result_text,
        ], alignment=ft.MainAxisAlignment.CENTER, expand=True),
    )


ft.app(target=main)