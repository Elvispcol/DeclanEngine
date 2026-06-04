# DeclanEngine
## Institutional Synthetic Indices Context Engine

Motor de interpretación institucional de mercado para índices sintéticos Boom/Crash de Deriv.

---

## Filosofía

> "El edge no está en encontrar más trades. El edge está en eliminar entornos de baja calidad."

Este sistema NO es un indicador tradicional. Es un **motor de interpretación contextual** basado en:
- Metodología Wyckoff
- Estructura de mercado institucional
- Comportamiento de liquidez
- Análisis de presión de velas
- Confirmación multitemporal

---

## Estructura del Proyecto

```
DeclanEngine/
├── structure_engine/
│   ├── __init__.py
│   ├── swing_detector.py        # Detección de swings (fractales)
│   ├── structure_classifier.py  # HH/HL/LH/LL
│   ├── bos_choch_detector.py    # BOS y CHOCH
│   └── displacement_detector.py # Desplazamiento / impulso
├── data/
│   └── sample_generator.py      # Generador de datos OHLC de prueba
├── tests/
│   └── test_structure.py        # Tests del motor de estructura
├── main.py                      # Punto de entrada
└── requirements.txt
```

---

## Sprints de Desarrollo

| Sprint | Módulo                  | Estado     |
|--------|-------------------------|------------|
| 1      | Structure Engine        | ✅ Activo  |
| 2      | Liquidity Engine        | 🔜 Pendiente |
| 3      | Candle Pressure Engine  | 🔜 Pendiente |
| 4      | Wyckoff Phase Engine    | 🔜 Pendiente |
| 5      | Probability Engine      | 🔜 Pendiente |
| 6      | MT5 Integration         | 🔜 Pendiente |
| 7      | Visual Overlay + Alerts | 🔜 Pendiente |

---

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/Elvispcol/DeclanEngine.git
cd DeclanEngine

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar motor de estructura
python main.py
```

---

## Requisitos

- Python 3.11+
- pandas
- numpy
- MetaTrader5 (Sprint 6)

---

## Marca

**Declan Trader** | Porciento Trading  
Motor desarrollado bajo doctrina institucional de interpretación de mercados sintéticos.
