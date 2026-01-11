import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import io

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Clasificador CACES IA")

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #4B5563;}
    .success-box {padding: 1rem; background-color: #D1FAE5; border-radius: 0.5rem; color: #065F46;}
    /* Ajuste para que el editor de datos ocupe buen espacio */
    .stDataFrame {width: 100%;}
</style>
""", unsafe_allow_html=True)

# --- BASE DE DATOS DE CONOCIMIENTO (CACES - ESTRUCTURA COMPLETA) ---
ESQUEMA_ACADEMICO = {
    "Medicina": {
        "Medicina Interna": {
            "Emergencias clínicas": ["Shock cardiogénico", "Shock hipovolémico", "Shock anafiláctico", "Shock séptico", "Síncope", "Soporte vital básico y avanzado"],
            "Sistema cardiovascular": ["Electrofisiología", "Síndrome coronario", "Insuficiencia cardiaca", "HTA", "Arritmias", "Valvulopatías", "Cor pulmonale"],
            "Sistema tegumentario": ["Manifestaciones cutáneas", "Acné", "Enf. degenerativas piel", "Dermatitis seborreica", "Micosis", "Pediculosis", "Piodermias", "Escabiosis", "Urticaria y angioedema"],
            "Aparato digestivo": ["ERGE", "Enfermedad ácido péptica", "Cáncer digestivo", "Hemorragia digestiva", "Diarrea aguda/crónica", "Estreñimiento", "Enf. inflamatoria intestinal", "Pancreatitis", "Hepatitis", "Cirrosis/Hipertensión portal", "Insuficiencia hepática"],
            "Sistema endócrino": ["Síndrome metabólico", "Dislipidemias", "Complicaciones glucosa", "Diabetes Mellitus 1 y 2", "Patologías tiroides", "Osteoporosis", "Adenomas hipofisiarios", "Patología suprarrenal"],
            "Sistema hematopoyético": ["Anemias y policitemias", "Hemoderivados", "Leucemias", "Linfomas"],
            "Enfermedades infecciosas": ["Fiebre origen desconocido", "Tétanos", "Celulitis/erisipela", "Varicela/Herpes", "ETS", "Tuberculosis", "Parasitosis", "Zoonosis", "VIH-SIDA", "Sepsis", "Fiebre reumática", "SARS-COV2", "Enfermedades tropicales (Dengue, Malaria, etc)"],
            "Aparato renal y urinario": ["Infecciones urinarias", "Insuficiencia renal aguda/crónica", "Síndrome nefrítico y nefrótico"],
            "Sistema nervioso": ["Equilibrio", "Cefalea", "Epilepsia/Convulsiones", "Encefalopatía", "ACV/ECV", "Infecciones SN", "Neuralgia trigémino", "Guillain Barré"],
            "Aparato respiratorio": ["Infecciones respiratorias altas/bajas", "Asma", "Derrame pleural", "EPOC", "Insuficiencia respiratoria", "Tromboembolia"],
            "Enfermedades autoinmunes": ["Lupus", "Artritis", "Espondilitis", "Esclerosis", "Sjogren"]
        },
        "Pediatría": {
            "Neonatología": ["Recepción RN", "Reanimación neonatal", "Displasia cadera", "Asfixia/Enf. hipóxico-isquémica", "Hipoglicemia", "Líquidos y electrolitos", "Ictericia", "Prematuridad/RCIU", "Sepsis neonatal", "SDR", "Malformaciones congénitas"],
            "Pediatría General": ["Hematología", "Imagenología", "Líquidos/electrolitos", "RCP pediátrico", "Accidentes", "Malnutrición", "Deshidratación", "Convulsión febril", "Síndrome metabólico", "Maltrato/Abuso", "Anemia", "Urticaria/Exantemas", "Infecciones piel", "IRA Altas/Bajas", "Soplos", "Asma", "AIEPI", "ERGE", "Diarrea/Parasitosis", "Patología testicular", "ITU", "Nefrítico/Nefrótico", "Crisis comiciales", "Infecciones SN", "Inmunizaciones (PAI)", "Nutrición/Lactancia", "COVID Pediátrico"]
        },
        "Gíneco Obstetricia": {
            "Ginecología": ["Climaterio/Osteoporosis", "Amenorrea", "Cáncer (mama, cérvix, endometrio, ovario)", "Leucorrea", "Dolor pélvico", "Dismenorrea", "Ciclo menstrual", "SOP", "Hemorragia uterina", "ITS", "Planificación familiar"],
            "Obstetricia": ["Aborto", "Hemorragia obstétrica", "Diagnóstico embarazo", "Control prenatal", "Embarazo múltiple", "Parto normal/anormal", "Trastornos hipertensivos (Preeclampsia)", "Parto pretérmino", "Incompatibilidad Rh/ABO", "Puerperio normal/patológico", "RCIU", "RPM", "Sufrimiento fetal", "Diabetes gestacional"]
        },
        "Cirugía": {
            "Cirugía general": ["Asepsia", "Heridas", "Infección sitio quirúrgico", "Líquidos", "Pre/Postoperatorio", "Profilaxis", "Quemaduras", "Trauma (Tórax, Abdomen, Craneal)", "Patología biliar", "Hernias"],
            "Abdomen agudo": ["Apendicitis", "Obstructivo", "Ano rectal"],
            "Oftalmología": ["Ametropías", "Conjuntivitis", "Estrabismo", "Glaucoma", "Uveitis", "Blefaritis", "Trauma ocular"],
            "Otorrinolaringología": ["Rinitis", "Amigdalitis", "Epistaxis", "Otitis", "Sinusitis", "Trauma nasal", "Vértigo"],
            "Traumatología": ["Luxaciones", "Túnel Carpiano", "Quervain", "Artrosis", "Escoliosis", "Esguinces", "Fracturas", "Lumbalgias", "Pie plano", "Osteomielitis", "Neoplasias óseas"],
            "Urología": ["Trauma testicular", "Balanitis", "Cáncer próstata", "Fimosis", "HPB", "Prostatitis", "Retención urinaria", "Torsión", "Varicocele", "Urolitiasis", "Uretritis"]
        },
        "Salud Mental": {
            "Condiciones psicosociales": ["Suicidio", "Alcohol y drogas", "Factores riesgo/protección"],
            "Trastornos mentales": ["Neurodesarrollo (Autismo, TDAH)", "Estado de ánimo (Depresión, Bipolar)", "Ansiedad", "Psicóticos (Esquizofrenia)", "Conducta alimentaria", "Neurocognitivos (Demencia)", "Adicciones"]
        },
        "Salud Pública": {
            "Atención primaria": ["Proceso salud-enfermedad", "Promoción/Prevención", "MAIS-FCI", "Grupos prioritarios", "Niveles de atención", "Gestión/ASIS", "Financiamiento"],
            "Epidemiología": ["Vigilancia epidemiológica", "Indicadores", "Medidas (Tasas, Riesgo)", "Determinación social", "Transmisibles/No transmisibles"],
            "Investigación": ["Bioestadística", "Tipos de estudio", "Metodología"],
            "Programas MSP": ["AIEPI", "PAI", "Nutrición", "Tuberculosis", "VIH-ITS", "Mortalidad materna", "Adulto mayor", "Adolescentes", "Violencia género"]
        },
        "Bioética": {
            "Bioética": ["Principios", "Dilemas (Vida/Muerte)", "Relación médico-paciente", "Consentimiento informado", "Ética investigación"]
        }
    },
    "Enfermería": {
        "Fundamentos del cuidado": {
            "Generalidades": ["Teorías (Nightingale, Orem, etc)", "Roles", "Pensamiento crítico"],
            "Procedimientos básicos": ["Higiene y confort", "Mecánica corporal", "Alimentación", "Eliminación", "Inmovilización", "Medicación", "Cuidados postmorten"],
            "Proceso de atención (PAE)": ["Valoración", "Taxonomías (NANDA, NOC, NIC)"],
            "Bioseguridad": ["Principios", "Limpieza/Esterilización", "Lavado manos", "Asepsia", "Desechos"],
            "Ética": ["Derechos paciente", "Código deontológico", "Aspectos legales (COIP)"],
            "Seguridad": ["Seguridad del paciente", "Prácticas seguras"],
            "Salud sexual": ["Anatomía reproductiva"]
        },
        "Cuidados mujer, RN, niño": {
            "Salud sexual mujer": ["Planificación", "Mortalidad materna", "Violencia"],
            "Embarazo, parto, puerperio": ["Control prenatal", "SCORE MAMA", "Complicaciones embarazo", "Parto", "Recién nacido sano", "Puerperio", "Lactancia materna"],
            "Gineco-obstétricos": ["Climaterio", "Cáncer ginecológico", "Cirugía ginecológica"],
            "Neonatología": ["Valoración RN", "Tamizaje", "Reanimación", "Termorregulación", "AIEPI Neonatal"],
            "Niñez y adolescencia": ["Crecimiento y desarrollo", "AIEPI Clínico", "Patologías prevalentes", "Inmunizaciones", "Problemas adolescencia"]
        },
        "Cuidados adulto y mayor": {
            "Generalidades": ["Gerontología", "Envejecimiento activo"],
            "Patologías clínicas": ["Respiratorias", "Cardiovasculares", "Metabólicas", "Neurológicas", "Digestivas", "Renales", "VIH", "Osteomusculares", "Vectores"],
            "Quirúrgico": ["Pre/Trans/Postoperatorio", "Heridas", "Ostomías"],
            "Procedimientos": ["Oxigenoterapia", "Insulina", "Sondas", "RCP básico"]
        },
        "Cuidado familiar/comunitario": {
            "Generalidades": ["MAIS-FCI", "Rol enfermera comunitaria"],
            "Bases cuidado": ["Determinantes salud", "Promoción", "Familia (Tipos, Ciclos)", "Comunidad"],
            "Trabajo familiar": ["Visita domiciliaria", "Ficha familiar", "ENI (Vacunas)", "Tuberculosis", "Epidemiología comunitaria"]
        },
        "Bases educativas/administrativas": {
            "Educación": ["Programas educativos", "Técnicas didácticas"],
            "Administración": ["Proceso administrativo", "Liderazgo", "Talento humano", "Calidad", "Registros"],
            "Investigación": ["Metodología", "Ética investigación"],
            "Epidemiología": ["Vigilancia", "Indicadores", "Brotes", "Bioestadística"]
        }
    },
    "Odontología": {
        "Operatoria dental": {
            "Lesiones cariosas": ["Etiología", "ICDAS", "Diagnóstico", "Tratamiento"],
            "Lesiones no cariosas": ["Etiología", "Clasificación", "Tratamiento"],
            "Procesos restauradores": ["Adhesión", "Técnicas directas"]
        },
        "Odontopediatría": {
            "Conducta": ["Manejo conducta niño"],
            "Desarrollo": ["Dentición", "Anomalías", "Defectos esmalte"],
            "Caries niño": ["Riesgo cariogénico", "Flúor", "Sellantes"],
            "Pulpa/Trauma": ["Terapia pulpar decidua", "Traumatismos", "Anestesia en niños"]
        },
        "Cirugía": {
            "Diagnóstico": ["Imagenología", "Exodoncia"],
            "Anestesia": ["Técnicas", "Complicaciones"],
            "Procedimientos": ["Cirugía preprotésica", "Infecciones", "Urgencias"]
        },
        "Rehabilitación Oral": {
            "Oclusión": ["ATM", "Tipos oclusión"],
            "Prótesis fija": ["Biomecánica", "Preparación", "Cementación"],
            "Prótesis removible": ["Clasificación Kennedy", "Diseño"],
            "Prótesis total": ["Retención", "Impresión"],
            "Endodonciados": ["Postes"]
        },
        "Endodoncia": {
            "Diagnóstico": ["Patología pulpar/periapical"],
            "Tratamiento": ["Preparación conductos", "Obturación"],
            "Complicaciones": ["Retratamiento", "Accidentes"]
        },
        "Periodoncia": {
            "Generalidades": ["Anatomía", "Etiopatogenia"],
            "Clasificación 2017": ["Salud", "Gingivitis", "Periodontitis"],
            "Tratamiento": ["Fases tratamiento periodontal"]
        },
        "Patología bucal": {
            "Tejidos duros": ["Quistes", "Tumores"],
            "Tejidos blandos": ["Lesiones blancas/rojas", "Cáncer oral", "Síndromes"]
        },
        "Farmacología": {
            "Anestésicos": ["Tipos", "Dosis", "Vasoconstrictores"],
            "Analgésicos/AINES": ["Mecanismo", "Dosis", "Interacciones"],
            "Antibióticos": ["Tipos", "Profilaxis", "Resistencia"]
        },
        "Medicina Interna": {
            "Manejo pacientes especiales": ["Diabetes", "Hipertensión", "Embarazo", "Anticoagulados", "Urgencias médicas en consultorio"]
        }
    }
}

# --- FUNCIONES ---

def configurar_api():
    """Configuración de la barra lateral"""
    with st.sidebar:
        st.header("⚙️ Configuración")
        api_key = st.text_input("Ingresa tu API Key de Google Gemini", type="password")
        st.info("Esta clave conecta la app con el cerebro de Google AI.")
        
        # Verificación del esquema cargado (Opcional para el usuario)
        with st.expander("Ver Esquema de Temas Cargado"):
            carrera = st.selectbox("Selecciona Carrera", list(ESQUEMA_ACADEMICO.keys()))
            st.json(ESQUEMA_ACADEMICO[carrera])
            
        return api_key

def procesar_con_ia(texto, api_key):
    """Lógica central de conexión con Gemini"""
    if not api_key: return "⚠️ Error: Falta ingresar la API Key."
    
    genai.configure(api_key=api_key)
    
    # --- SOLUCIÓN AL ERROR 404 ---
    # Intentamos conectar con el modelo más nuevo. Si falla, usamos el clásico.
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        # Hacemos una prueba rápida de conexión
        model.generate_content("test") 
    except:
        # Si el modelo flash falla o no existe, usamos el modelo Pro estándar
        model = genai.GenerativeModel("gemini-pro")
    
    # Instrucciones maestras para la IA
    prompt = f"""
    Actúa como un Experto Académico y evaluador oficial del examen CACES.
    
    TU MISIÓN:
    Analiza el texto proporcionado que contiene preguntas de examen.
    1. Identifica la respuesta correcta (si no está marcada, dedúcela por conocimiento médico).
    2. Genera un feedback educativo breve justificando la respuesta.
    3. CLASIFICACIÓN ESTRICTA: Usa EXCLUSIVAMENTE el siguiente esquema JSON para asignar Carrera, Componente, Subcomponente y Tema.
    
    ESQUEMA OFICIAL:
    {json.dumps(ESQUEMA_ACADEMICO, ensure_ascii=False)}

    FORMATO DE SALIDA REQUERIDO:
    Devuelve SOLAMENTE una lista de objetos JSON válida (Array).
    [
        {{
            "Pregunta": "Texto completo de la pregunta...",
            "Opciones de Respuesta": "A)... B)...",
            "Respuesta correcta": "La opción correcta",
            "feedback": "Explicación breve...",
            "Carrera": "Medicina/Enfermería/Odontología",
            "Componente": "Según esquema",
            "Subcomponente": "Según esquema",
            "Tema": "Según esquema"
        }}
    ]
    
    TEXTO A PROCESAR: 
    {texto}
    """
    
    try:
        response = model.generate_content(prompt)
        # Limpieza de la respuesta para obtener solo el JSON puro
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        return f"Error procesando la solicitud: {str(e)}. Intenta con menos preguntas a la vez."

def convertir_excel(df):
    """Convierte el DataFrame a Excel para descargar"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Banco_Preguntas')
        # Ajustar ancho de columnas
        worksheet = writer.sheets['Banco_Preguntas']
        for i, col in enumerate(df.columns):
            width = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, min(width, 50))
    return output.getvalue()

# --- INTERFAZ DE USUARIO (UI) ---

st.markdown('<div class="main-header">🎓 Gestor de Preguntas CACES</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clasificación Automática con IA</div>', unsafe_allow_html=True)

api_key = configurar_api()

# Pestañas para elegir modo de uso
tab1, tab2 = st.tabs(["📝 Pegar Texto Manualmente", "📂 Subir Archivo Excel"])
data_a_procesar = None

with tab1:
    txt_input = st.text_area("Pega aquí tus preguntas (aunque estén desordenadas):", height=200)
    if st.button("Procesar Texto", type="primary"): 
        data_a_procesar = txt_input

with tab2:
    file = st.file_uploader("Sube tu Excel (.xlsx)", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
        st.write("Vista previa:", df.head(2))
        col = st.selectbox("¿En qué columna está el texto de la pregunta?", df.columns)
        if st.button("Procesar Excel", type="primary"):
            # Unimos todas las preguntas en un solo texto para enviarlas a la IA
            data_a_procesar = "\n---\n".join(df[col].astype(str).tolist())

# --- ZONA DE RESULTADOS ---

if data_a_procesar:
    if not api_key:
        st.error("⚠️ Por favor ingresa tu API Key en el menú de la izquierda.")
    else:
        with st.status("🤖 La IA está trabajando...", expanded=True) as status:
            st.write("Analizando contenido médico...")
            st.write("Clasificando según temario CACES...")
            
            resultado = procesar_con_ia(data_a_procesar, api_key)
            
            if isinstance(resultado, list):
                status.update(label="¡Proceso Completado!", state="complete", expanded=False)
                
                df_res = pd.DataFrame(resultado)
                
                st.divider()
                st.subheader("✅ Revisa y Edita los Resultados")
                
                # Editor interactivo tipo Excel
                edited_df = st.data_editor(
                    df_res, 
                    num_rows="dynamic",
                    use_container_width=True
                )
                
                st.divider()
                
                # Botón de Descarga
                excel_bytes = convertir_excel(edited_df)
                st.download_button(
                    label="📥 Descargar Excel Final (.xlsx)",
                    data=excel_bytes,
                    file_name="preguntas_caces_procesadas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            else:
                status.update(label="Error", state="error")
                st.error("Hubo un problema con la respuesta de la IA:")
                st.warning(resultado)
