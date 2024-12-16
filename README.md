# AleBot - Trading Bot com Machine Learning

Um bot de trading avançado para Bitcoin que utiliza Machine Learning e análise técnica para maximizar lucros e minimizar riscos.

## Características

- Deep Learning para previsão de mercado
- Análise técnica automatizada
- Interface gráfica em tempo real
- Gerenciamento de risco integrado
- Backtesting e validação de estratégias
- Otimização automática de parâmetros

## Estrutura do Projeto

```
AleBot/
├── Core/                      # Núcleo do sistema
│   ├── TradingCore/          # Lógica principal de trading
│   ├── DataManager/          # Gerenciamento de dados
│   └── logger.py             # Sistema de logging
│
├── TradingSystem/            # Sistema de trading
│   ├── Core/                 # Componentes principais
│   ├── Interface/            # Interface gráfica
│   └── Config/               # Configurações
│
└── models/                   # Modelos treinados
```

## Requisitos

- Python 3.8+
- PyTorch
- Dash
- pandas
- numpy
- ta-lib
- python-binance

## Instalação

1. Clone o repositório:
```bash
git clone <seu-repositorio>
cd AleBot
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure suas credenciais da Binance em `config.py`

4. Execute o sistema:
```bash
python TradingSystem/main.py
```

## Configuração

1. Defina suas chaves API da Binance
2. Ajuste os parâmetros de trading em Settings
3. Configure os limites de risco
4. Selecione o modelo de ML desejado

## Uso

1. Inicie o sistema através da interface web
2. Monitore as previsões em tempo real
3. Acompanhe o desempenho do modelo
4. Ajuste parâmetros conforme necessário

## Segurança

- Nunca compartilhe suas chaves API
- Mantenha backups regulares
- Use limites de risco apropriados
- Monitore o sistema regularmente

## Contribuição

1. Fork o projeto
2. Crie sua branch de feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes. 