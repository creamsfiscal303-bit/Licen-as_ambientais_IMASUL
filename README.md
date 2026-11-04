# Extrator IMASUL — Plantio florestal e cruzamento cadastral

Substitua o `app.py` atual por este arquivo e mantenha a pasta `dados_imasul` existente para preservar os checkpoints.

Dependências em `requirements.txt`. O mapa foi temporariamente retirado desta versão.

O filtro de classes é conservador e mantém apenas descrições que indiquem explicitamente plantio florestal, floresta plantada, silvicultura ou reflorestamento. O cruzamento cadastral é feito por ID do empreendimento e, na ausência de correspondência, por número exato do processo. CPF não é presumido: só é preenchido se a camada publicar esse campo.
