import os
import cv2
from flask import Flask, request, render_template, redirect, url_for
from deepface import DeepFace
import matplotlib
import shutil

matplotlib.use('Agg')  # Usar o backend Agg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def limpar_pasta_static():
    pasta_static = 'static'
    if os.path.exists(pasta_static):
        shutil.rmtree(pasta_static)  # Remove a pasta e todo o seu conteúdo
    os.makedirs(pasta_static, exist_ok=True)  # Recria a pasta vazia


# Funções de processamento de imagem já definidas aqui (como no código anterior)
def detectar_rostos(imagem):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5), gray


def processar_rosto(face_image):
    # Como definido anteriormente
    return face_image  # Retorne a imagem processada


def analisar_emocoes(faces):
    resultados = []
    for i in range(1, len(faces) + 1):
        img_path = f"{app.config['UPLOAD_FOLDER']}/rosto_{i}.jpg"
        img2 = cv2.imread(img_path)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        resultado = DeepFace.analyze(img2, actions=['emotion'], enforce_detection=False)

        # Gerar o gráfico de emoções
        if resultado:
            gerar_grafico_emocoes(resultado[0]['emotion'], i, img_path)

        resultados.append(resultado)

    return resultados


def gerar_grafico_emocoes(emocoes, indice, img_path):
    import matplotlib.pyplot as plt

    emoções_nomes = list(emocoes.keys())
    emoções_valores = list(emocoes.values())

    plt.figure(figsize=(10, 5))
    plt.bar(emoções_nomes, emoções_valores, color='skyblue')
    plt.title('Distribuição das Emoções')
    plt.xlabel('Emoções')
    plt.ylabel('Intensidade (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Salvar a figura
    plt.tight_layout()
    plt.savefig(f'{app.config["UPLOAD_FOLDER"]}/grafico_emocoes_{indice}.png')
    plt.close()  # Fechar a figura para liberar recursos


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    imagem = cv2.imread(filepath)
    faces, gray = detectar_rostos(imagem)

    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            face_image = gray[y:y + h, x:x + w]
            cv2.imwrite(f"{app.config['UPLOAD_FOLDER']}/rosto_{i + 1}.jpg", face_image)

        resultados = analisar_emocoes(faces)
        return render_template('result.html', resultados=resultados)
    else:
        return render_template('result.html', resultados=[])


if __name__ == "__main__":
    limpar_pasta_static()
    app.run(debug=True)