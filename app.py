# Importando as bibliotecas necessárias
from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import os

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

        # Criando gráficos individuais para cada cluster
        img_files = create_cluster_plots(df_clustered)

        return render_template('result.html', img_files=img_files)

# Função para aplicar o K-Means
def apply_kmeans(df, n_clusters):
    data_for_clustering = df[['horas_ouvidas_rock', 'horas_ouvidas_samba', 'horas_ouvidas_pop', 'horas_ouvidas_rap']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(data_for_clustering)
    return df[['horas_ouvidas_rock', 'horas_ouvidas_samba', 'horas_ouvidas_pop', 'horas_ouvidas_rap', 'cluster']]

# Função para criar gráficos individuais para cada cluster e salvar imagens
def create_cluster_plots(df):
    img_files = []
    static_path = os.path.join(app.root_path, 'static')

    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]

        # Selecionando o estilo fixo (por exemplo, rock) para o eixo x
        fixed_style = 'horas_ouvidas_rock'

        # Selecionando os outros estilos musicais para o eixo y
        other_styles = ['horas_ouvidas_samba', 'horas_ouvidas_pop', 'horas_ouvidas_rap']

        scatter_matrix = px.scatter_matrix(cluster_data, dimensions=[fixed_style] + other_styles,
                                           color='cluster', title=f'Perfil de Usuários - Cluster {cluster}')

        img_name = f'scatter_matrix_cluster{cluster}.png'
        img_files.append(img_name)
        img_path = os.path.join(static_path, img_name)
        scatter_matrix.write_image(img_path)

    return img_files

# Executar a aplicação Flask
if __name__ == '__main__':
    app.run(debug=True)
