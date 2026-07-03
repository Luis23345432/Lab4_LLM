import json
from student_agent import AssemblyAgent
from evaluator import calcular_score_plan

ARCHIVO_DESARROLLO = "Examples.json" 

def qwen_lazy(**kwargs):
    from llm_engine import qwen
    return qwen(**kwargs)

def main(n_casos=10):
    print(f"Cargando dataset de desarrollo: {ARCHIVO_DESARROLLO}")
    with open(ARCHIVO_DESARROLLO, 'r') as f:
        casos = json.load(f)
        
    agente = AssemblyAgent()
    puntaje_total = 0.0
    casos_evaluados = min(n_casos, len(casos)) # Limite para pruebas rapidas
    
    print("-" * 50)
    for i in range(casos_evaluados):
        caso = casos[i]
        print(f"Evaluando Tarea ID: {caso['assembly_task_id']} (Longitud optima: {caso['complexity_level']})")
        
        plan_generado = agente.solve(caso['scenario_context'], qwen_lazy)
        plan_optimo = caso['target_action_sequence']
        
        # Calculo de metrica
        score = calcular_score_plan(plan_generado, plan_optimo)
        puntaje_total += score
        
        print(f"Plan Generado: {plan_generado}")
        print(f"Score obtenido: {score} / 10.0\n")
        
    promedio = puntaje_total / casos_evaluados
    print("-" * 50)
    print(f"Puntaje Promedio en Desarrollo: {round(promedio, 2)} / 10.0")

if __name__ == "__main__":
    main(10)
