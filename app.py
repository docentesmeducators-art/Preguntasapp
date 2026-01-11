import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import io
import os
import pypdf
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Clasificador CACES IA")

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #4B5563;}
    .success-box {padding: 1rem; background-color: #D1FAE5; border-radius: 0.5rem; color: #065F46;}
    .stDataFrame {width: 100%;}
</style>
""", unsafe_allow_html=True)

# --- BASE DE DATOS DE CONOCIMIENTO (CACES - ESTRUCTURA NUMERADA OFICIAL) ---
ESQUEMA_ACADEMICO = {
    "Medicina": {
        "1. Medicina Interna": [
            "1.1 Emergencias clínicas", "1.2. Sistema cardiovascular", "1.3. Sistema tegumentario", 
            "1.4. Aparato digestivo", "1.5. Sistema endócrino", "1.6. Sistema hematopoyético", 
            "1.7. Enfermedades infecciosas", "1.8. Aparato renal y urinario", "1.9. Sistema nervioso", 
            "1.10. Aparato respiratorio", "1.11. Enfermedades autoinmunes"
        ],
        "2. Pediatría": [
            "2.1. Neonatología", "2.2. Pediatría"
        ],
        "3. Gíneco Obstetricia": [
            "3.1. Ginecología", "3.2. Obstetricia"
        ],
        "4. Cirugía": [
            "4.1 Cirugía general", "4.2 Abdomen agudo", "4.3 Oftalmología", 
            "4.4 Otorrinolaringología", "4.5 Traumatología", "4.6 Urología"
        ],
        "5. Salud Mental": [
            "5.1. Condiciones psicosociales por ciclos de vida", "5.2. Trastornos mentales"
        ],
        "6. Salud Pública": [
            "6.1. Componentes de atención primaria de salud", "6.2. Epidemiología", 
            "6.3. Investigación en salud", "6.4 Programas y estrategias del Ministerio de Salud Pública"
        ],
        "7. Bioética": [
            "7.1. Bioética"
        ]
    },
    "Enfermería": {
        "1. Fundamentos del cuidado enfermero": [
            "1.1. Generalidades para el cuidado enfermero", "1.2. Procedimientos básicos del cuidado enfermero",
            "1.3. Proceso de atención en Enfermería", "1.4. Bioseguridad", "1.5. Ética en el ejercicio profesional",
            "1.6. Seguridad y calidad en el cuidado enfermero", "1.7. Salud sexual y reproductiva"
        ],
        "2. Cuidados de la mujer, recién nacido, niño y adolescente": [
            "2.1. Salud sexual y reproductiva de la mujer", "2.2. Cuidados de enfermería en el embarazo, parto y puerperio",
            "2.3. Cuidados gíneco obstétricos de la mujer", "2.4. Cuidados de enfermería en el recién nacido",
            "2.5. Generalidades sobre niñez y adolescencia", "2.6. Cuidados de enfermería en la niñez y adolescencia"
        ],
        "3. Cuidados del adulto y adulto mayor": [
            "3.1. Generalidades del cuidado de enfermería del adulto y adulto mayor",
            "3.2. Cuidados de enfermería en el adulto y adulto mayor",
            "3.3. Cuidados de enfermería a personas con problemas quirúrgicos más frecuentes",
            "3.4. Procedimientos básicos del cuidado enfermero en pacientes adultos y adultos mayores"
        ],
        "4. Cuidado familiar, comunitario e intercultural": [
            "4.1. Generalidades sobre el cuidado familiar y comunitario", "4.2. Bases para el cuidado familiar y comunitario",
            "4.3. La enfermería en el trabajo familiar y comunitario"
        ],
        "5. Bases educativas, administrativas, investigativas y epidemiológicas del cuidado enfermero": [
            "5.1. Educación para la salud", "5.2. Bases administrativas del cuidado",
            "5.3. Bases de investigación científica: metodología de investigación",
            "5.4. Bases epidemiológicas del cuidado: vigilancia epidemiológica"
        ]
    },
    "Odontología": {
        "1. Operatoria dental": [
            "1.1. Lesiones cariosas", "1.2. Lesiones no cariosas", "1.3. Procesos restauradores directos"
        ],
        "2. Odontopediatría": [
            "2.1. Técnicas de manejo de la conducta de pacientes pediátricos", "2.2. Desarrollo dental y anomalías del desarrollo",
            "2.3. Higiene oral mecánica y química en el hogar", "2.4. Caries dental en el niño y el adolescente",
            "2.5. Selladores de fosas y fisuras y uso de fluoruros", "2.6. Alteraciones pulpares en dientes deciduos y control del dolor",
            "2.7. Traumatismos de los dientes y tejidos de sostén"
        ],
        "3. Cirugía": [
            "3.1. Diagnóstico clínico y complementario", "3.2. Anestesia", "3.3. Principios de la técnica quirúrgica y exodoncia",
            "3.4. Indicaciones para cirugía pre protésica", "3.5. Infecciones bucales y maxilares", "3.6. Manejo de urgencias en cirugía"
        ],
        "4. Rehabilitación Oral": [
            "4.1. Oclusión", "4.2. Prótesis fija", "4.3. Prótesis parcial removible", "4.4 Prótesis total", 
            "4.5. Rehabilitación de dientes endodonciados"
        ],
        "5. Endodoncia": [
            "5.1. Diagnóstico de alteraciones pulpares y periapicales", "5.2. Tratamiento endodóncico",
            "5.3. Retratamiento y cirugía periapical", "5.4. Complicaciones en endodoncia"
        ],
        "6. Periodoncia": [
            "6.1. Anatomía periodontal", "6.2. Exámenes diagnósticos en periodoncia",
            "6.3. Etiopatogenia de la enfermedad periodontal", 
            "6.4. Diagnóstico y clasificación de patologías periodontales (clasificación 2017)",
            "6.5. Tratamiento periodontal"
        ],
        "7. Patología Bucal": [
            "7.1. Lesiones de tejidos duros y diagnóstico diferencial", "7.2. Patología de tejidos blandos y diagnóstico diferencial",
            "7.3 Síndromes sistémicos"
        ],
        "8. Farmacología": [
            "8.1. Anestésicos locales", "8.2. Analgésicos y antiinflamatorios", 
            "8.3. Antibacterianos", "8.4. Antivirales y antimicóticos"
        ],
        "9. Medicina Interna": [
            "9.1. Enfermedades metabólicas", "9.2. Enfermedades respiratorias", "9.3. Embarazo",
            "9.4. Urgencias y emergencias en Odontología", "9.5. Enfermedades cardiovasculares",
            "9.6. Trastornos de la coagulación", "9.7. Soporte vital básico"
        ]
    }
}

# --- GESTIÓN DE BIBLIOTECA (SISTEMA DE ARCHIVOS) ---
DIRECTORIO_BASE = "biblioteca_digital"

def inicializar_carpetas():
    if not os.path.exists(DIRECTORIO_BASE):
        os.makedirs(DIRECTORIO_BASE)
    for carrera in ESQUEMA_ACADEMICO.keys():
        ruta = os.path.join(DIRECTORIO_BASE, carrera)
        if not os.path.exists(ruta):
            os.makedirs(ruta)

def guardar_pdf(archivo, carrera):
    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carrera)
    ruta_archivo = os.path.join(ruta_carpeta, archivo.name)
    with open(ruta_archivo, "wb") as f:
        f.write(archivo.getbuffer())
    return ruta_archivo

def listar_archivos(carrera):
    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carrera)
    if os.path.exists(ruta_carpeta):
        return [f for f in os.listdir(ruta_carpeta) if f.endswith('.pdf')]
    return []

def leer_biblioteca_carrera(carrera):
    texto_total = ""
    archivos = listar_archivos(carrera)
    lista_fuentes = []
    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carrera)
    
    for nombre_archivo in archivos:
        try:
            ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
            reader = pypdf.PdfReader(ruta_completa)
            texto_archivo = f"\n--- INICIO FUENTE: {nombre_archivo} ---\n"
            for page in reader.pages[:50]: 
                texto_archivo += page.extract_text() + "\n"
            texto_archivo += f"\n--- FIN FUENTE: {nombre_archivo} ---\n"
            texto_total += texto_archivo
            lista_fuentes.append(nombre_archivo)
        except Exception as e:
            print(f"Error leyendo {nombre_archivo}: {e}")
            
    return texto_total, lista_fuentes

# --- FUNCIONES DE IA ---

def configurar_api():
    with st.sidebar:
        st.header("⚙️ Configuración")
        api_key = st.text_input("Ingresa tu API Key de Google Gemini", type="password")
        st.divider()
        st.write("📚 **Estado de la Biblioteca**")
        inicializar_carpetas()
        for carrera in ESQUEMA_ACADEMICO.keys():
            n = len(listar_archivos(carrera))
            st.caption(f"- {carrera}: {n} documentos")
        return api_key

def autodetectar_modelo(api_key):
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available_models: return None, "No hay modelos disponibles."
        modelo_elegido = next((m for m in available_models if 'flash' in m.lower()), None)
        if not modelo_elegido:
            modelo_elegido = next((m for m in available_models if 'pro' in m.lower()), available_models[0])
        return genai.GenerativeModel(modelo_elegido), None
    except Exception as e:
        return None, str(e)

def procesar_con_ia(texto, api_key, carrera_seleccionada):
    if not api_key: return "⚠️ Error: Falta API Key."
    texto_bibliografia, fuentes = leer_biblioteca_carrera(carrera_seleccionada)
    model, error = autodetectar_modelo(api_key)
    if error: return f"Error IA: {error}"
    
    contexto_extra = ""
    if texto_bibliografia:
        contexto_extra = f"""
        URGENTE - USA ESTA BIBLIOGRAFÍA OFICIAL:
        Documentos cargados: {', '.join(fuentes)}.
        Prioriza esta información para las respuestas y feedback.
        
        CONTENIDO BIBLIOTECA:
        {texto_bibliografia[:300000]} 
        """
    
    prompt = f"""
    Actúa como un Evaluador Académico CACES (Ecuador).
    
    {contexto_extra}
    
    TAREA:
    Analiza las preguntas proporcionadas.
    
    1. **CORRECCIÓN DE FORMA (PERMITIDO)**: Si la pregunta original tiene errores ortográficos, dobles espacios, falta de tildes o saltos de línea que dificultan la lectura, CORRÍGELOS para que se vea profesional.
    2. **CORRECCIÓN DE FONDO (PROHIBIDO)**: NO cambies la terminología médica, los valores clínicos ni el sentido de la pregunta.
    
    REGLAS ESTRICTAS DE FORMATO Y CLASIFICACIÓN:
    1. **Opciones**: 4 opciones separadas por "|".
    2. **Respuesta Correcta**: COPIA EXACTA e IDÉNTICA de la opción correcta.
    3. **Feedback**: Estructura OBLIGATORIA con saltos de línea:
       - Respuesta correcta: [Explicación]
       - Respuestas incorrectas: [Explicación]
       - Mnemotecnia/Tip: [Opcional]
       - Bibliografía: [CITA OBLIGATORIA EN FORMATO VANCOUVER]
    4. **Clasificación**: Debes usar los nombres EXACTOS del siguiente esquema, incluyendo sus NÚMEROS (ej: "1.1 Emergencias clínicas").
    
    ESQUEMA OFICIAL ({carrera_seleccionada}):
    {json.dumps(ESQUEMA_ACADEMICO[carrera_seleccionada], ensure_ascii=False)}

    SALIDA JSON (Array):
    [
        {{
            "Pregunta": "Texto corregido (solo forma)...",
            "Opciones de Respuesta": "...",
            "Respuesta correcta": "...",
            "feedback": "...",
            "Carrera": "{carrera_seleccionada}",
            "Componente": "...",
            "Subcomponente": "...",
            "Tema": "..."
        }}
    ]
    
    PREGUNTAS A PROCESAR: 
    {texto}
    """
    
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        return f"Error procesando: {str(e)}"

def convertir_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Banco_Preguntas')
        worksheet = writer.sheets['Banco_Preguntas']
        for i, col in enumerate(df.columns):
            width = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, min(width, 50))
    return output.getvalue()

# --- INTERFAZ UI ---

inicializar_carpetas()
api_key = configurar_api()

st.title("🎓 Gestor Académico Inteligente")

modo = st.radio("Selecciona una opción:", ["📝 Procesar Preguntas", "📚 Administrar Biblioteca"], horizontal=True)

if modo == "📚 Administrar Biblioteca":
    st.header("Gestor de Documentos")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Subir Nuevo Documento")
        carrera_upload = st.selectbox("¿A qué carrera pertenece el libro/guía?", list(ESQUEMA_ACADEMICO.keys()))
        archivo_pdf = st.file_uploader("Sube el PDF aquí", type=["pdf"])
        if archivo_pdf and st.button("Guardar en Biblioteca", type="primary"):
            ruta = guardar_pdf(archivo_pdf, carrera_upload)
            st.success(f"✅ Guardado en: {carrera_upload}")
            st.balloons()
            time.sleep(1)
            st.rerun()

    with col2:
        st.subheader("Documentos Existentes")
        for carrera in ESQUEMA_ACADEMICO.keys():
            with st.expander(f"📂 {carrera}"):
                archivos = listar_archivos(carrera)
                if archivos:
                    for f in archivos:
                        st.markdown(f"📄 {f}")
                else:
                    st.caption("Carpeta vacía")

elif modo == "📝 Procesar Preguntas":
    st.header("Procesamiento de Exámenes")
    col_config, col_input = st.columns([1, 2])
    with col_config:
        st.info("Configuración de Contexto")
        carrera_proceso = st.selectbox("¿De qué carrera son estas preguntas?", list(ESQUEMA_ACADEMICO.keys()))
        libros_disponibles = listar_archivos(carrera_proceso)
        if libros_disponibles:
            st.success(f"✅ {len(libros_disponibles)} fuentes disponibles.")
        else:
            st.warning("⚠️ Sin bibliografía específica. Usando conocimiento general.")

    with col_input:
        tab_text, tab_file = st.tabs(["Pegar Texto", "Subir Excel"])
        texto_final = None
        with tab_text:
            txt = st.text_area("Pega las preguntas aquí:", height=150)
            if st.button("Procesar Texto"): texto_final = txt
        with tab_file:
            file = st.file_uploader("Sube Excel", type=["xlsx"])
            if file:
                df = pd.read_excel(file)
                c = st.selectbox("Columna Pregunta", df.columns)
                if st.button("Procesar Excel"): 
                    texto_final = "\n---\n".join(df[c].astype(str).tolist())

    if texto_final:
        with st.status("🧠 Analizando con Biblioteca...", expanded=True) as status:
            res = procesar_con_ia(texto_final, api_key, carrera_proceso)
            if isinstance(res, list):
                status.update(label="¡Completado!", state="complete", expanded=False)
                df_res = pd.DataFrame(res)
                st.divider()
                st.subheader("Resultados")
                editado = st.data_editor(df_res, num_rows="dynamic", use_container_width=True)
                st.download_button("📥 Descargar Excel", convertir_excel(editado), "banco_preguntas.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
            else:
                st.error("Error:")
                st.warning(res)
