# 🤟 Rockit Vision — AI Hand Gesture Recognition

Um sistema web de visão computacional em tempo real para reconhecimento, classificação e tradução de gestos manuais, desenvolvido como Trabalho de Conclusão de Curso (TCC) em Sistemas de Informação.

O projeto utiliza a webcam do usuário diretamente no navegador para capturar frames, processar a malha esquelética das mãos e classificar o gesto exibido em tempo real através de uma arquitetura baseada em WebSockets.

## 🚀 Funcionalidades

* **Reconhecimento em Tempo Real:** Captura e processamento de vídeo de alta performance utilizando a câmera nativa do dispositivo.
* **Comunicação via WebSockets:** Envio de frames e recebimento de predições sem recarregar a página, garantindo fluidez e baixo *delay*.
* **Processamento de Malha (Landmarks):** Extração precisa das coordenadas das articulações da mão usando MediaPipe.
* **Classificação de Gestos:** Suporte atual para detecção de múltiplos sinais, incluindo: *Coração, Hangloose, Joinha, Olá, Paz, Rock e Spock*.
* **Interface Responsiva:** UI moderna construída com FastHTML, incluindo controle de qualidade de imagem e métricas de FPS ao vivo.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.12
* **Visão Computacional & IA:** OpenCV, MediaPipe (Tasks Vision), Scikit-Learn, Joblib
* **Backend & Servidor:** FastHTML, Uvicorn, Starlette, WebSockets
* **Infraestrutura:** Docker (com dependências nativas `libgl1`, `libgles2`, `libegl1`)
* **Deploy:** Render (PaaS) com suporte a HTTPS nativo

## 📂 Estrutura do Projeto

* `/app.py`: Arquivo principal contendo as rotas web e os endpoints do WebSocket.
* `/core/`: Módulo do motor da aplicação (`processor.py`, `utils.py`, `models.py`, etc).
* `/models/`: Modelos de Machine Learning treinados (`.joblib` e `.task`) e codificadores de labels.
* `/assets/`: Arquivos estáticos (CSS, JavaScript, e imagens de referência dos gestos).
* `Dockerfile`: Receita de infraestrutura para construção da imagem Linux e deploy.
* `pyproject.toml`: Gerenciador de pacotes e metadados do projeto.

## 💻 Como Rodar Localmente

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
   cd SEU_REPOSITORIO