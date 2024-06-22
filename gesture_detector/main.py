# pylint: disable=C0116, C0114, W0401
from imports import *  # NOSONAR

def load_image(image_name: str):
    return mediapipe.Image.create_from_file(image_name)

def create_ai():
    base = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    opts = vision.GestureRecognizerOptions(base_options=base)
    return vision.GestureRecognizer.create_from_options(opts)

def detect_gesture(ai, image):
    output = ai.recognize(image)
    return output.gestures[0][0].score, output.gestures[0][0].category_name

def print_result(accuracy, gesture):
    print(f'Gesture is "{gesture}" with an accuracy of {round(accuracy * 100, 2)}%')


def main(image_name):
    image = load_image(image_name)
    ai = create_ai()
    accuracy, gesture = detect_gesture(ai, image)
    print_result(accuracy, gesture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='image name')
    args = parser.parse_args()
    main(args.filename)
