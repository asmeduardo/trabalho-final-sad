# Importando as bibliotecas necessárias
from flask import Flask, render_template, request
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.cluster import KMeans
import json

# Inicializando a aplicação Flask
app = Flask(__name__)

# Rota para a página de upload
@app.route('/')
def upload_page():
    return render_template('upload.html')

# Rota para processar o arquivo enviado
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('upload.html', error='Nenhum arquivo enviado')

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', error='Nome do arquivo vazio')

    if file:
        # Lendo o arquivo CSV
        df = pd.read_csv(file, delimiter=';')

        # Aplicando o K-Means
        df_clustered = apply_kmeans(df, n_clusters=8)

        # Gerando subplots e salvando as imagens
        generate_subplots(df_clustered)

        return render_template('result.html', plot='Clusters processados com sucesso!')

# Função para aplicar o K-Means
def apply_kmeans(df, n_clusters):
    data_for_clustering = df[['horas_ouvidas_rock', 'horas_ouvidas_samba', 'horas_ouvidas_pop', 'horas_ouvidas_rap']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(data_for_clustering)
    return df

# Função para gerar subplots e salvar imagens
def generate_subplots(df):
    size_factor = 0.5
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Rock', 'Samba', 'Pop', 'Rap'))

    for i, estilo in enumerate(['horas_ouvidas_rock', 'horas_ouvidas_samba', 'horas_ouvidas_pop', 'horas_ouvidas_rap']):
        row = (i // 2) + 1
        col = (i % 2) + 1

        scatter = px.scatter(df, x=estilo, y='cluster', size='horas_ouvidas_pop', opacity=0.7,
                              labels={estilo: f'Horas de {estilo.capitalize()}', 'cluster': 'Cluster'},
                              title=f'Perfil de Usuários - {estilo.capitalize()} (Clusters: 8)')

        scatter.update_traces(marker=dict(size=df['horas_ouvidas_pop'] * size_factor))

        # Adicionando a imagem do subplot ao layout
        fig.add_trace(scatter['data'][0], row=row, col=col)

        # Salvando a imagem individualmente
        img_name = f'dispersao_{estilo}_clusters.png'
        scatter.write_image(img_name)

    # Atualizando o layout
    fig.update_layout(height=800, width=1000, title_text="Perfil de Usuários - Estilos Musicais (Clusters: 8)")

    # Salvando a imagem do layout
    img_name = 'dispersao_clusters.png'
    fig.write_image(img_name)

# Executar a aplicação Flask
if __name__ == '__main__':
    app.run(debug=True)
