from __future__ import annotations

import json
import re
from functools import partial
from typing import Any

import cv2
import gradio as gr
import numpy as np
from openai import OpenAI


client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

SYSTEM_PROMPT = """
Ты преобразуешь команду пользователя в JSON для обработки изображения.
Верни только один JSON-объект без markdown и пояснений.

Доступные действия:
1. {"action": "rotate", "angle": 90}  # допустимо 90, 180, 270
2. {"action": "resize", "scale": 0.5}
3. {"action": "grayscale"}
4. {"action": "extract_channel", "channel": "R"}  # R, G или B
5. {"action": "blur", "kernel": 15}  # нечётное число

Если команда неясна, верни {"action": "unknown"}.
"""

SUPPORTED_ACTIONS = {"rotate", "resize", "grayscale", "extract_channel", "blur"}
QUICK_COMMANDS: list[tuple[str, dict[str, Any]]] = [
    ("Повернуть 90°", {"action": "rotate", "angle": 90}),
    ("Повернуть 180°", {"action": "rotate", "angle": 180}),
    ("Ч/Б", {"action": "grayscale"}),
    ("Красный канал", {"action": "extract_channel", "channel": "R"}),
    ("Зелёный канал", {"action": "extract_channel", "channel": "G"}),
    ("Синий канал", {"action": "extract_channel", "channel": "B"}),
    ("Размытие 9", {"action": "blur", "kernel": 9}),
    ("Уменьшить 50%", {"action": "resize", "scale": 0.5}),
]

TEXT_EXAMPLES = [
    "Сделай изображение чёрно-белым",
    "Поверни картинку на 90 градусов",
    "Покажи только красный канал",
    "Слегка размой изображение",
    "Уменьши изображение в 2 раза",
]


def _safe_json_loads(raw_content: str) -> dict[str, Any]:
    content = raw_content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def _image_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def parse_command_with_llm(user_instruction: str) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_instruction},
            ],
            temperature=0.1,
        )
        raw_content = response.choices[0].message.content or ""
        data = _safe_json_loads(raw_content)
        return data if isinstance(data, dict) else {"action": "unknown"}
    except Exception as exc:
        print(f"LLM error: {exc}")
        return {"action": "unknown"}


def parse_command_locally(user_instruction: str) -> dict[str, Any]:
    text = user_instruction.lower()

    if any(token in text for token in ("чб", "черно-бел", "чёрно-бел", "grayscale", "gray")):
        return {"action": "grayscale"}

    if "красн" in text or "red" in text:
        return {"action": "extract_channel", "channel": "R"}
    if "зел" in text or "green" in text:
        return {"action": "extract_channel", "channel": "G"}
    if "син" in text or "blue" in text:
        return {"action": "extract_channel", "channel": "B"}

    if "разм" in text or "blur" in text:
        kernel_match = re.search(r"(\d+)", text)
        kernel = int(kernel_match.group(1)) if kernel_match else 9
        if kernel % 2 == 0:
            kernel += 1
        return {"action": "blur", "kernel": kernel}

    if "повер" in text or "rotate" in text:
        angle_match = re.search(r"(90|180|270)", text)
        return {"action": "rotate", "angle": int(angle_match.group(1)) if angle_match else 90}

    if "уменьш" in text or "resize" in text or "масштаб" in text:
        if "2" in text:
            return {"action": "resize", "scale": 0.5}
        if "3" in text:
            return {"action": "resize", "scale": 0.33}
        return {"action": "resize", "scale": 0.75}

    return {"action": "unknown"}


def parse_command(user_instruction: str) -> dict[str, Any]:
    local_command = parse_command_locally(user_instruction)
    if local_command.get("action") != "unknown":
        return local_command
    return parse_command_with_llm(user_instruction)


def apply_command(image: np.ndarray | None, command_data: dict[str, Any]) -> np.ndarray | None:
    if image is None:
        return None

    action = command_data.get("action")
    if action not in SUPPORTED_ACTIONS:
        return _image_to_rgb(_image_to_bgr(image))

    img = _image_to_bgr(image)

    if action == "rotate":
        angle = int(command_data.get("angle", 90))
        if angle == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif action == "resize":
        scale = float(command_data.get("scale", 0.5))
        scale = max(scale, 0.05)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    elif action == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif action == "extract_channel":
        channel = str(command_data.get("channel", "R")).upper()
        b, g, r = cv2.split(img)
        zeros = np.zeros_like(b)
        if channel == "R":
            img = cv2.merge([zeros, zeros, r])
        elif channel == "G":
            img = cv2.merge([zeros, g, zeros])
        elif channel == "B":
            img = cv2.merge([b, zeros, zeros])

    elif action == "blur":
        kernel = int(command_data.get("kernel", 9))
        if kernel < 3:
            kernel = 3
        if kernel % 2 == 0:
            kernel += 1
        img = cv2.GaussianBlur(img, (kernel, kernel), 0)

    return _image_to_rgb(img)


def process_text_instruction(image: np.ndarray | None, instruction: str):
    if image is None or not instruction.strip():
        return image
    command_data = parse_command(instruction)
    print(f"Executing: {command_data}")
    return apply_command(image, command_data)


def process_quick_action(image: np.ndarray | None, command_data: dict[str, Any]):
    if image is None:
        return None
    print(f"Quick command: {command_data}")
    return apply_command(image, command_data)


def clear_instruction():
    return "", None


with gr.Blocks(title="AI Управление OpenCV") as demo:
    gr.Markdown(
        """
        # AI Управление OpenCV
        Загрузи изображение, напиши команду или нажми одну из быстрых кнопок.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Исходное изображение", type="numpy")
            instruction = gr.Textbox(
                label="Команда",
                placeholder="Например: сделай чёрно-белым",
            )
            with gr.Row():
                run_button = gr.Button("Выполнить", variant="primary")
                clear_button = gr.Button("Очистить")
            gr.Examples(examples=TEXT_EXAMPLES, inputs=instruction, label="Примеры команд")

        with gr.Column(scale=1):
            result = gr.Image(label="Результат")
            gr.Markdown("### Быстрые команды")
            for row_start in range(0, len(QUICK_COMMANDS), 2):
                with gr.Row():
                    for label, command_data in QUICK_COMMANDS[row_start : row_start + 2]:
                        button = gr.Button(label)
                        button.click(
                            fn=partial(process_quick_action, command_data=command_data),
                            inputs=image_input,
                            outputs=result,
                        )

    run_button.click(fn=process_text_instruction, inputs=[image_input, instruction], outputs=result)
    instruction.submit(fn=process_text_instruction, inputs=[image_input, instruction], outputs=result)
    clear_button.click(fn=clear_instruction, inputs=None, outputs=[instruction, result])


if __name__ == "__main__":
    print("Запускаю веб-интерфейс. Открой ссылку, которая появится ниже.")
    demo.launch()
