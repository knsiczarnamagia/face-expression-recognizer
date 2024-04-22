# Face Expression Recognizer

Face Expression Recognizer is an application which uses custom machine learning algorithms trained for analysing video and image data.
It's main goal is to detect and classify human face expressions.

The project is the result of work done by members of the Czarna Magia AI Student Club.

## :rocket: Demo

The project is hosted on Hugging Face Spaces :hugs:. Please, feel free to check out our
:sparkles: [live demo app on Gradio](https://huggingface.co/spaces/jlynxdev/face-expression-recognizer)! :sparkles:

## Dataset

The project's AI models have been trained on the dataset which comes from the
[Facial Expression Recognition Challenge on Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview).

## Tech Stack

- Python => 3.11
- PyTorch (`torch` and `torchvision`)
- OpenCV
- Matplotlib
- Weights & Biases

## How to run this app

1. Clone the repository: First, you need to clone the repository to your local machine. You can do this with the `git clone` command:
    ```
    git clone https://github.com/knsiczarnamagia/face-expression-recognizer.git
    cd face-expression-recognizer
    ```

2. Install packages: You have two options to install the required packages listed in `requirements.txt`:
    - Globally: You can install the packages globally using `pip`. Run the following command in your terminal:
        ```
        pip install -r requirements.txt
        ```
    - Using a virtual environment: You can also create a virtual environment and install the packages there. This is recommended as it avoids installing packages globally and keeps dependencies for this project separate. Here’s how you can do it:
        - Conda:
        ```
        conda create --name face-expression-recognizer
        conda activate face-expression-recognizer
        pip install -r requirements.txt
        ```
        - venv:
        ```
        python -m venv face-expression-recognizer
        source face-expression-recognizer/bin/activate
        pip install -r requirements.txt
        ```
3. Run the application: After installing the packages, you can run the application by executing `app.py` in your console. Use the following command:
    ```
    python app.py
    ```
    After running this command, a URL should be displayed in the console. This is the URL where your application is running and can be accessed.
4. Access the application: Paste displayed URL in the console to your browser.

## Authors and acknowledgement

:clap: The project has been created by the following developers:
- Jakub Hryniewicz ([KubaHryna](https://github.com/KubaHryna))
- Dawid Koterwas ([Kiwinicki](https://github.com/Kiwinicki))
- Jacek Nowak ([jlynxdev](https://github.com/jlynxdev))
- Remigiusz Sęk ([remigiuszsek](https://github.com/remigiuszsek))