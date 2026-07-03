# Laboratorio 4: Planning LLM

Solucion para el laboratorio de planificacion con Qwen3-8B. No se entrena ni se hace fine-tuning del modelo.

## Archivos principales

- `student_agent.py`: agente principal. Detecta el dominio, genera planes canonicos y contiene un parser robusto para respuestas de Qwen.
- `llm_engine.py`: carga `Qwen/Qwen3-8B` en 4 bits y usa inferencia deterministica.
- `dev_test.py`: evalua los primeros casos de `Examples.json`.
- `submit.py`: procesa `Task.json` y genera `submission.json`.
- `evaluator.py`: calcula el score por longitud exacta, prefijo correcto y match completo.

## Requisitos

En Colab, instala las dependencias necesarias para `transformers`, `accelerate`, `bitsandbytes` y `torch`.

La inferencia usa:

- `MODEL_ID = "Qwen/Qwen3-8B"`
- `temperature=0.0`
- `do_sample=False`
- `enable_thinking=False`

## Probar

```bash
python dev_test.py
```

## Generar entrega

```bash
python submit.py
```

El agente usa un planificador deterministico para los dominios conocidos y mantiene Qwen3-8B como fallback deterministico si un caso no encaja con esos dominios.

El archivo `submission.json` se genera como una lista de objetos con:

- `assembly_task_id`
- `complexity_level`
- `target_action_sequence`
