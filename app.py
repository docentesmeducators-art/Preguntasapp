import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import io
import os
import pypdf
import time  # <--- ESTA ES LA LÍNEA QUE FALTABA

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

# --- BASE DE DATOS DE CONOCIMIENTO (CACES) ---
ESQUEMA_ACADEMICO = {
    "Medicina": {
        "Medicina Interna": ["Emergencias clínicas", "Sistema cardiovascular", "Sistema tegumentario", "Aparato digestivo", "Sistema endócrino", "Sistema hematopoyético", "Enfermedades infecciosas", "Aparato renal y urinario", "Sistema nervioso", "Aparato respiratorio", "Enfermedades autoinmunes"],
        "Pediatría": ["Neonatología", "Pediatría General"],
        "Gíneco Obstetricia": ["Ginecología", "Obstetricia"],
        "Cirugía": ["Cirugía general", "Abdomen agudo", "Oftalmología", "Otorrinolaringología", "Traumatología", "Urología"],
        "Salud Mental": ["Condiciones psicosociales", "Trastornos mentales"],
        "Salud Pública": ["Atención primaria", "Epidemiología", "Investigación", "Programas MSP"],
        "Bioética": ["Bioética"]
    },
    "Enfermería": {
        "Fundamentos del cuidado": ["Generalidades", "Procedimientos básicos", "Proceso de atención (PAE)", "Bioseguridad", "Ética", "Seguridad", "Salud sexual"],
        "Cuidados mujer, RN, niño": ["Salud sexual mujer", "Embarazo, parto, puerperio", "Gineco-obstétricos", "Neonatología", "Niñez y adolescencia"],
        "Cuidados adulto y mayor": ["Generalidades", "Patologías clínicas", "Quirúrgico", "Procedimientos"],
        "Cuidado familiar/comunitario": ["Generalidades", "Bases cuidado", "Trabajo familiar"],
        "Bases educativas/administrativas": ["Educación", "Administración", "Investigación", "Epidemiología"]
    },
    "Odontología": {
        "Operatoria dental": ["Lesiones cariosas", "Lesiones no cariosas", "Procesos restauradores"],
        "Odontopediatría": ["Conducta", "Desarrollo", "Caries niño", "Pulpa/Trauma"],
        "Cirugía": ["Diagnóstico", "Anestesia", "Procedimientos"],
        "Rehabilitación Oral": ["Oclusión", "Prótesis fija", "Prótesis removible", "Prótesis total", "Endodonciados"],
        "Endodoncia": ["Diagnóstico", "Tratamiento", "Complicaciones"],
        "Periodoncia": ["Generalidades", "Clasificación 2017", "Tratamiento"],
        "Patología bucal": ["Tejidos duros", "Tejidos blandos"],
        "Farmacología": ["Anestésicos", "Analgésicos/AINES", "Antibióticos"],
        "Medicina Interna": ["Manejo pacientes especiales"]
    }
}

# --- GESTIÓN DE BIBLIOTECA (SISTEMA DE ARCHIVOS) ---
DIRECTORIO_BASE = "biblioteca_digital"

def inicializar_carpetas():
    """Crea las carpetas si no existen"""
    if not os.path.exists(DIRECTORIO_BASE):
        os.makedirs(DIRECTORIO_BASE)
    
    for carrera in ESQUEMA_ACADEMICO.keys():
        ruta = os.path.join(DIRECTORIO_BASE, carrera)
        if not os.path.exists(ruta):
            os.makedirs(ruta)

def guardar_pdf(archivo, carrera):
    """Guarda un archivo subido en la carpeta correspondiente"""
    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carrera)
    ruta_archivo = os.path.join(ruta_carpeta, archivo.name)
    
    with open(ruta_archivo, "wb") as f:
        f.write(archivo.getbuffer())
    return ruta_archivo

def listar_archivos(carrera):
    """Devuelve la lista de PDFs guardados para una carrera"""
    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carrera)
    if os.path.exists(ruta_carpeta):
        return [f for f in os.listdir(ruta_carpeta) if f.endswith('.pdf')]
    return []

def leer_biblioteca_carrera(carrera):
    """Lee todos los PDFs de una carrera y extrae su texto"""
    texto_total = ""
    archivos = listar_archivos(carrera)
    lista_fuentes = []
    
    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carrera)
    
    for nombre_archivo in archivos:
        try:
            ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
            reader = pypdf.PdfReader(ruta_completa)
            texto_archivo = f"\n--- INICIO FUENTE: {nombre_archivo} ---\n"
            # Limitamos a las primeras 50 páginas por libro para no saturar memoria
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
        
        # Monitor de Biblioteca
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
        
        # Prioridad: Flash > Pro
        modelo_elegido = next((m for m in available_models if 'flash' in m.lower()), None)
        if not modelo_elegido:
            modelo_elegido = next((m for m in available_models if 'pro' in m.lower()), available_models[0])
            
        return genai.GenerativeModel(modelo_elegido), None
    except Exception as e:
        return None, str(e)

def procesar_con_ia(texto, api_key, carrera_seleccionada):
    if not api_key: return "⚠️ Error: Falta API Key."
    
    # 1. Cargar conocimiento de la biblioteca local
    texto_bibliografia, fuentes = leer_biblioteca_carrera(carrera_seleccionada)
    
    # 2. Configurar IA
    model, error = autodetectar_modelo(api_key)
    if error: return f"Error IA: {error}"
    
    # 3. Construir Prompt
    contexto_extra = ""
    if texto_bibliografia:
        contexto_extra = f"""
        URGENTE - USA ESTA BIBLIOGRAFÍA OFICIAL PARA RESPONDER:
        El usuario ha proporcionado documentos oficiales ({', '.join(fuentes)}).
        Tu máxima prioridad es basar las respuestas y el feedback en estos textos.
        
        CONTENIDO BIBLIOTECA LOCAL:
        {texto_bibliografia[:300000]} 
        """
    
    prompt = f"""
    Actúa como un Evaluador Académico CACES (Ecuador).
    
    {contexto_extra}
    
    TAREA:
    Analiza las preguntas, estandarízalas y clasifícalas.
    
    REGLAS ESTRICTAS DE FORMATO:
    1. **Opciones**: 4 opciones separadas por "|". (Ej: "A|B|C|D").
    2. **Correcta**: COPIA EXACTA de una de las opciones.
    3. **Feedback**: Estructura OBLIGATORIA con saltos de línea:
       - Respuesta correcta: [Explicación]
       - Respuestas incorrectas: [Explicación]
       - Mnemotecnia/Tip: [Opcional]
       - Bibliografía: [CITA EL DOCUMENTO DE LA BIBLIOTECA USADO]
    
    ESQUEMA DE CLASIFICACIÓN ({carrera_seleccionada}):
    {json.dumps(ESQUEMA_ACADEMICO[carrera_seleccionada], ensure_ascii=False)}

    SALIDA JSON (Array):
    [
        {{
            "Pregunta": "...",
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

# Navegación Principal
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
            st.success(f"✅ Archivo guardado correctamente en: {carrera_upload}")
            st.balloons()
            time.sleep(1)
            st.rerun()

    with col2:
        st.subheader("Documentos Existentes")
        st.info("Estos son los libros que la IA leerá automáticamente.")
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
        
        # Mostrar qué libros se usarán
        libros_disponibles = listar_archivos(carrera_proceso)
        if libros_disponibles:
            st.success(f"✅ Se usarán {len(libros_disponibles)} fuentes de la biblioteca de {carrera_proceso}.")
        else:
            st.warning(f"⚠️ La carpeta de {carrera_proceso} está vacía. La IA usará conocimiento general.")

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
                
                st.download_button(
                    "📥 Descargar Excel",
                    convertir_excel(editado),
                    "banco_preguntas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            else:
                st.error("Error:")
                st.warning(res)
