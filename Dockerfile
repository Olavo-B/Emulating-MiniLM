# Use uma imagem base do Python
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Configura diretório de trabalho no container
WORKDIR /app

# Cria um usuário não-root
RUN useradd -m appuser

# Copia o código do host para o container
COPY . /app

# Ajusta permissões para o novo usuário
RUN chown -R appuser /app

# Alterna para o usuário não-root
USER appuser

# Instala dependências
RUN pip install --user --no-cache-dir -r requirements.txt

# Comando padrão
CMD ["bash"]
