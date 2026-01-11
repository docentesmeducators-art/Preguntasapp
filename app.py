import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.api_core import retry
import json
import io
import os
import pypdf
import time
import re

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Clasificador CACES IA")

# --- CSS PARA MEJORAR LA VISUALIZACIÓN ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; font-weight: bold;}
    .sub-header {font-size: 1.5rem; color: #4B5563;}
    .stTextArea textarea {font-size: 14px;}
    /* Ajuste para que las celdas de feedback se vean mejor con saltos de línea */
    div[data-testid="stDataFrame"] div[role="grid"] div[role="row"] {
        min-height: 100px !important; 
    }
</style>
""", unsafe_allow_html=True)

# --- BASE DE DATOS MAESTRA (HARDCODED) ---
ESQUEMA_ACADEMICO = {
    "Medicina": {
        "1. Medicina Interna": {
            "1.1 Emergencias clínicas": [
                "1.1.1 Shock cardiogénico", "1.1.2 Shock hipovolémico", "1.1.3 Shock anafiláctico", 
                "1.1.4 Shock séptico", "1.1.5 Síncope", "1.1.6 Soporte vital básico y avanzado"
            ],
            "1.2. Sistema cardiovascular": [
                "1.2.1 Electrofisiología", "1.2.2 Síndrome coronario agudo", "1.2.3 Insuficiencia cardiaca", 
                "1.2.4 Hipertensión arterial", "1.2.5 Arritmias", "1.2.6 Valvulopatías", "1.2.7 Cor pulmonale"
            ],
            "1.3. Sistema tegumentario": [
                "1.3.1 Manifestaciones cutáneas de enfermedades sistémicas", "1.3.2 Acné", "1.3.3 Enfermedades degenerativas de la piel",
                "1.3.4 Dermatitis seborreica", "1.3.5 Micosis superficiales", "1.3.6 Pediculosis", "1.3.7 Piodermias",
                "1.3.8 Escabiosis", "1.3.9 Urticaria y angioedema"
            ],
            "1.4. Aparato digestivo": [
                "1.4.1 Enfermedad por reflujo gastroesofágico", "1.4.2 Enfermedad ácido péptica", "1.4.3 Cáncer digestivo",
                "1.4.4 Hemorragia digestiva", "1.4.5 Diarrea aguda y crónica", "1.4.6 Estreñimiento", 
                "1.4.7 Enfermedad inflamatoria intestinal", "1.4.8 Pancreatitis", "1.4.9 Hepatitis", 
                "1.4.10 Cirrosis e Hipertensión portal", "1.4.11 Insuficiencia hepática"
            ],
            "1.5. Sistema endócrino": [
                "1.5.1 Síndrome metabólico", "1.5.2 Dislipidemias", "1.5.3 Complicaciones agudas y crónicas por alteraciones de la glucosa",
                "1.5.4 Diabetes Mellitus tipo 1 y 2", "1.5.5 Patologías de tiroides", "1.5.6 Osteoporosis",
                "1.5.7 Adenomas hipofisiarios", "1.5.8 Patología de las glándulas suprarrenales"
            ],
            "1.6. Sistema hematopoyético": [
                "1.6.1 Anemias y policitemias", "1.6.2 Hemoderivados", "1.6.3 Leucemias", "1.6.4 Linfomas"
            ],
            "1.7. Enfermedades infecciosas": [
                "1.7.1 Fiebre de origen desconocido", "1.7.2 Tétanos", "1.7.3 Infecciones de piel y partes blandas",
                "1.7.4 Varicela y Herpes zóster", "1.7.5 Infecciones de transmisión sexual", "1.7.6 Tuberculosis",
                "1.7.7 Parasitosis", "1.7.8 Zoonosis", "1.7.9 VIH-SIDA", "1.7.10 Sepsis", "1.7.11 Fiebre reumática",
                "1.7.12 SARS-COV2", "1.7.13 Enfermedades tropicales y metaxénicas"
            ],
            "1.8. Aparato renal y urinario": [
                "1.8.1 Infecciones de vías urinarias", "1.8.2 Insuficiencia renal aguda y crónica", 
                "1.8.3 Síndrome nefrítico y nefrótico"
            ],
            "1.9. Sistema nervioso": [
                "1.9.1 Trastornos del equilibrio", "1.9.2 Cefalea", "1.9.3 Epilepsia y síndromes convulsivos",
                "1.9.4 Encefalopatía", "1.9.5 Enfermedad cerebro vascular", "1.9.6 Infecciones del sistema nervioso central",
                "1.9.7 Neuralgia del trigémino", "1.9.8 Síndrome de Guillain Barré"
            ],
            "1.10. Aparato respiratorio": [
                "1.10.1 Infecciones respiratorias altas y bajas", "1.10.2 Asma", "1.10.3 Derrame pleural",
                "1.10.4 Enfermedad pulmonar obstructiva crónica", "1.10.5 Insuficiencia respiratoria", "1.10.6 Tromboembolia pulmonar"
            ],
            "1.11. Enfermedades autoinmunes": [
                "1.11.1 Lupus eritematoso sistémico", "1.11.2 Artritis reumatoide", "1.11.3 Espondilitis anquilosante",
                "1.11.4 Esclerosis sistémica", "1.11.5 Síndrome de Sjogren"
            ]
        },
        "2. Pediatría": {
            "2.1. Neonatología": [
                "2.1.1 Recepción del recién nacido", "2.1.2 Reanimación neonatal", "2.1.3 Displasia de cadera",
                "2.1.4 Asfixia y encefalopatía hipóxico-isquémica", "2.1.5 Hipoglicemia neonatal", "2.1.6 Líquidos y electrolitos",
                "2.1.7 Ictericia neonatal", "2.1.8 Prematuridad y RCIU", "2.1.9 Sepsis neonatal", "2.1.10 Dificultad respiratoria del recién nacido",
                "2.1.11 Malformaciones congénitas"
            ],
            "2.2. Pediatría": [
                "2.2.1 Hematología", "2.2.2 Imagenología", "2.2.3 Líquidos y electrolitos", "2.2.4 Reanimación cardiopulmonar pediátrica",
                "2.2.5 Accidentes y violencia", "2.2.6 Malnutrición", "2.2.7 Deshidratación", "2.2.8 Convulsión febril",
                "2.2.9 Síndrome metabólico", "2.2.10 Maltrato y abuso sexual", "2.2.11 Anemia", "2.2.12 Urticaria y exantemas",
                "2.2.13 Infecciones de la piel", "2.2.14 Infecciones respiratorias altas y bajas", "2.2.15 Soplos cardíacos",
                "2.2.16 Asma", "2.2.17 AIEPI Clínico", "2.2.18 Reflujo gastroesofágico", "2.2.19 Diarrea y parasitosis",
                "2.2.20 Patología testicular y escrotal", "2.2.21 Infección de vías urinarias", "2.2.22 Síndrome nefrítico y nefrótico",
                "2.2.23 Crisis comiciales", "2.2.24 Infecciones del sistema nervioso", "2.2.25 Inmunizaciones (PAI)",
                "2.2.26 Nutrición y lactancia materna", "2.2.27 COVID Pediátrico"
            ]
        },
        "3. Gíneco Obstetricia": {
            "3.1. Ginecología": [
                "3.1.1 Climaterio y osteoporosis", "3.1.2 Amenorrea", "3.1.3 Cáncer de mama, cérvix, endometrio y ovario",
                "3.1.4 Leucorrea", "3.1.5 Dolor pélvico crónico", "3.1.6 Dismenorrea", "3.1.7 Ciclo menstrual",
                "3.1.8 Síndrome de ovario poliquístico", "3.1.9 Hemorragia uterina anormal", "3.1.10 Infecciones de transmisión sexual",
                "3.1.11 Planificación familiar"
            ],
            "3.2. Obstetricia": [
                "3.2.1 Aborto", "3.2.2 Hemorragia obstétrica", "3.2.3 Diagnóstico de embarazo", "3.2.4 Control prenatal",
                "3.2.5 Embarazo múltiple", "3.2.6 Parto normal y anormal", "3.2.7 Trastornos hipertensivos del embarazo",
                "3.2.8 Parto pretérmino", "3.2.9 Incompatibilidad Rh y ABO", "3.2.10 Puerperio normal y patológico",
                "3.2.11 Restricción del crecimiento intrauterino", "3.2.12 Ruptura prematura de membranas",
                "3.2.13 Sufrimiento fetal agudo y crónico", "3.2.14 Diabetes gestacional"
            ]
        },
        "4. Cirugía": {
            "4.1 Cirugía general": [
                "4.1.1 Asepsia y antisepsia", "4.1.2 Manejo de heridas", "4.1.3 Infección de sitio quirúrgico",
                "4.1.4 Manejo de líquidos y electrolitos", "4.1.5 Manejo pre y postoperatorio", "4.1.6 Profilaxis antibiótica",
                "4.1.7 Quemaduras", "4.1.8 Trauma de tórax, abdomen y craneoencefálico", "4.1.9 Patología biliar", "4.1.10 Hernias de pared abdominal"
            ],
            "4.2 Abdomen agudo": [
                "4.2.1 Apendicitis aguda", "4.2.2 Abdomen agudo obstructivo", "4.2.3 Patología ano rectal benigna"
            ],
            "4.3 Oftalmología": [
                "4.3.1 Ametropías", "4.3.2 Conjuntivitis", "4.3.3 Estrabismo", "4.3.4 Glaucoma", "4.3.5 Uveitis",
                "4.3.6 Blefaritis", "4.3.7 Trauma ocular"
            ],
            "4.4 Otorrinolaringología": [
                "4.4.1 Rinitis", "4.4.2 Amigdalitis", "4.4.3 Epistaxis", "4.4.4 Otitis", "4.4.5 Sinusitis",
                "4.4.6 Trauma nasal", "4.4.7 Vértigo"
            ],
            "4.5 Traumatología": [
                "4.5.1 Luxación congénita de cadera", "4.5.2 Síndrome del túnel carpiano", "4.5.3 Tenosinovitis de Quervain",
                "4.5.4 Artrosis", "4.5.5 Escoliosis", "4.5.6 Esguinces", "4.5.7 Fracturas y luxaciones", "4.5.8 Lumbalgias",
                "4.5.9 Pie plano", "4.5.10 Osteomielitis", "4.5.11 Neoplasias óseas"
            ],
            "4.6 Urología": [
                "4.6.1 Trauma testicular", "4.6.2 Balanitis", "4.6.3 Cáncer de próstata", "4.6.4 Fimosis",
                "4.6.5 Hiperplasia prostática benigna", "4.6.6 Prostatitis", "4.6.7 Retención urinaria",
                "4.6.8 Torsión testicular", "4.6.9 Varicocele e hidrocele", "4.6.10 Urolitiasis", "4.6.11 Uretritis"
            ]
        },
        "5. Salud Mental": {
            "5.1. Condiciones psicosociales por ciclos de vida": [
                "5.1.1 Suicidio", "5.1.2 Consumo problemático de alcohol y drogas", "5.1.3 Factores de riesgo y de protección"
            ],
            "5.2. Trastornos mentales": [
                "5.2.1 Trastornos del neurodesarrollo", "5.2.2 Trastornos del estado de ánimo", "5.2.3 Trastornos de ansiedad",
                "5.2.4 Trastornos psicóticos", "5.2.5 Trastornos de la conducta alimentaria", "5.2.6 Trastornos neurocognitivos",
                "5.2.7 Trastornos debidos al consumo de sustancias"
            ]
        },
        "6. Salud Pública": {
            "6.1. Componentes de atención primaria de salud": [
                "6.1.1 Proceso salud-enfermedad", "6.1.2 Promoción y prevención de salud", "6.1.3 MAIS-FCI",
                "6.1.4 Grupos prioritarios y vulnerables", "6.1.5 Niveles de atención, organización y funcionamiento",
                "6.1.6 Gestión y administración de salud (ASIS)", "6.1.7 Financiamiento del sistema de salud"
            ],
            "6.2. Epidemiología": [
                "6.2.1 Vigilancia epidemiológica", "6.2.2 Indicadores de salud", "6.2.3 Medidas de frecuencia, asociación e impacto",
                "6.2.4 Determinación social de la salud", "6.2.5 Enfermedades transmisibles y no transmisibles"
            ],
            "6.3. Investigación en salud": [
                "6.3.1 Bioestadística", "6.3.2 Tipos de estudio", "6.3.3 Metodología de la investigación"
            ],
            "6.4 Programas y estrategias del Ministerio de Salud Pública": [
                "6.4.1 Estrategia AIEPI", "6.4.2 Estrategia nacional de inmunizaciones (PAI)", "6.4.3 Nutrición saludable",
                "6.4.4 Control de tuberculosis", "6.4.5 Atención y control de VIH - ITS",
                "6.4.6 Reducción de la mortalidad materna y neonatal", "6.4.7 Salud integral del adulto mayor",
                "6.4.8 Salud integral de adolescentes", "6.4.9 Salud sexual y reproductiva", "6.4.10 Atención a víctimas de violencia"
            ]
        },
        "7. Bioética": {
            "7.1. Bioética": [
                "7.1.1 Definiciones de ética, moral, deontología, derechos y principios", "7.1.2 Principios de bioética",
                "7.1.3 Dilemas bioéticos: inicio y final de la vida", "7.1.4 Relación médico-paciente",
                "7.1.5 Consentimiento informado", "7.1.6 Ética de la investigación en salud"
            ]
        }
    },
    "Enfermería": {
        "1. Fundamentos del cuidado enfermero": {
            "1.1. Generalidades para el cuidado enfermero": ["1.1.1 Teorías y modelos de enfermería", "1.1.2 Roles de enfermería", "1.1.3 Pensamiento crítico"],
            "1.2. Procedimientos básicos del cuidado enfermero": ["1.2.1 Higiene y confort", "1.2.2 Mecánica corporal", "1.2.3 Alimentación", "1.2.4 Eliminación", "1.2.5 Medidas de inmovilización", "1.2.6 Administración de medicamentos", "1.2.7 Cuidados postmorten"],
            "1.3. Proceso de atención en Enfermería": ["1.3.1 Valoración de enfermería", "1.3.2 Taxonomías NANDA, NOC, NIC"],
            "1.4. Bioseguridad": ["1.4.1 Principios de bioseguridad", "1.4.2 Limpieza, desinfección y esterilización", "1.4.3 Lavado de manos", "1.4.4 Asepsia y antisepsia", "1.4.5 Manejo de desechos"],
            "1.5. Ética en el ejercicio profesional": ["1.5.1 Derechos del paciente", "1.5.2 Código deontológico", "1.5.3 Aspectos legales (COIP)"],
            "1.6. Seguridad y calidad en el cuidado enfermero": ["1.6.1 Seguridad del paciente", "1.6.2 Prácticas seguras"],
            "1.7. Salud sexual y reproductiva": ["1.7.1 Anatomía y fisiología del sistema reproductivo"]
        },
        "2. Cuidados de la mujer, recién nacido, niño y adolescente": {
            "2.1. Salud sexual y reproductiva de la mujer": ["2.1.1 Planificación familiar", "2.1.2 Mortalidad materna", "2.1.3 Violencia contra la mujer"],
            "2.2. Cuidados de enfermería en el embarazo, parto y puerperio": ["2.2.1 Control prenatal", "2.2.2 SCORE MAMA", "2.2.3 Complicaciones del embarazo", "2.2.4 Parto", "2.2.5 Atención inmediata del recién nacido", "2.2.6 Puerperio", "2.2.7 Lactancia materna"],
            "2.3. Cuidados gíneco obstétricos de la mujer": ["2.3.1 Climaterio y menopausia", "2.3.2 Cáncer ginecológico", "2.3.3 Cirugía ginecológica"],
            "2.4. Cuidados de enfermería en el recién nacido": ["2.4.1 Valoración del recién nacido", "2.4.2 Tamizaje neonatal", "2.4.3 Reanimación neonatal", "2.4.4 Termorregulación", "2.4.5 AIEPI Neonatal"],
            "2.5. Generalidades sobre niñez y adolescencia": ["2.5.1 Crecimiento y desarrollo", "2.5.2 Derechos de la niñez"],
            "2.6. Cuidados de enfermería en la niñez y adolescencia": ["2.6.1 AIEPI Clínico", "2.6.2 Patologías prevalentes", "2.6.3 Inmunizaciones", "2.6.4 Problemas de salud en la adolescencia"]
        },
        "3. Cuidados del adulto y adulto mayor": {
            "3.1. Generalidades del cuidado de enfermería del adulto y adulto mayor": ["3.1.1 Gerontología y geriatría", "3.1.2 Envejecimiento activo"],
            "3.2. Cuidados de enfermería en el adulto y adulto mayor": ["3.2.1 Patologías respiratorias", "3.2.2 Patologías cardiovasculares", "3.2.3 Patologías metabólicas", "3.2.4 Patologías neurológicas", "3.2.5 Patologías digestivas", "3.2.6 Patologías renales", "3.2.7 VIH/SIDA", "3.2.8 Patologías osteomusculares", "3.2.9 Enfermedades vectoriales"],
            "3.3. Cuidados de enfermería a personas con problemas quirúrgicos más frecuentes": ["3.3.1 Cuidados pre, trans y postoperatorios", "3.3.2 Cuidado de heridas", "3.3.3 Manejo de ostomías"],
            "3.4. Procedimientos básicos del cuidado enfermero en pacientes adultos y adultos mayores": ["3.4.1 Oxigenoterapia", "3.4.2 Administración de insulina", "3.4.3 Manejo de sondas", "3.4.4 RCP básico"]
        },
        "4. Cuidado familiar, comunitario e intercultural": {
            "4.1. Generalidades sobre el cuidado familiar y comunitario": ["4.1.1 MAIS-FCI", "4.1.2 Rol de la enfermera comunitaria"],
            "4.2. Bases para el cuidado familiar y comunitario": ["4.2.1 Determinantes de la salud", "4.2.2 Promoción de la salud", "4.2.3 Familia (Tipos, Ciclos)", "4.2.4 Comunidad"],
            "4.3. La enfermería en el trabajo familiar y comunitario": ["4.3.1 Visita domiciliaria", "4.3.2 Ficha familiar", "4.3.3 Estrategia Nacional de Inmunizaciones", "4.3.4 Control de tuberculosis", "4.3.5 Epidemiología comunitaria"]
        },
        "5. Bases educativas, administrativas, investigativas y epidemiológicas del cuidado enfermero": {
            "5.1. Educación para la salud": ["5.1.1 Programas educativos", "5.1.2 Técnicas didácticas"],
            "5.2. Bases administrativas del cuidado": ["5.2.1 Proceso administrativo", "5.2.2 Liderazgo", "5.2.3 Gestión del talento humano", "5.2.4 Calidad en salud", "5.2.5 Registros de enfermería"],
            "5.3. Bases de investigación científica: metodología de investigación": ["5.3.1 Metodología de la investigación", "5.3.2 Ética en la investigación"],
            "5.4. Bases epidemiológicas del cuidado: vigilancia epidemiológica": ["5.4.1 Métodos: epidemiológico y clínico", "5.4.2 Indicadores epidemiológicos y socioeconómicos del Ecuador", "5.4.3 Generalidades de medición", "5.4.4 Cálculo de proporciones, tasas y razones", "5.4.5 Sistema de vigilancia epidemiológica", "5.4.6 Tipos de vigilancia epidemiológica", "5.4.7 Investigación de brotes"]
        }
    },
    "Odontología": {
        "1. Operatoria dental": {
            "1.1. Lesiones cariosas": ["1.1.1 Etiología de la caries", "1.1.2 Sistema ICDAS", "1.1.3 Diagnóstico de caries", "1.1.4 Tratamiento de caries"], 
            "1.2. Lesiones no cariosas": ["1.2.1 Etiología", "1.2.2 Clasificación", "1.2.3 Tratamiento"], 
            "1.3. Procesos restauradores directos": ["1.3.1 Adhesión dental", "1.3.2 Técnicas directas de restauración"]
        },
        "2. Odontopediatría": {
            "2.1. Técnicas de manejo de la conducta de pacientes pediátricos": ["2.1.1 Farmacológicas", "2.1.2 No farmacológicas"], 
            "2.2. Desarrollo dental y anomalías del desarrollo": ["2.2.1 Cronología de la erupción", "2.2.2 Anomalías de forma, número y tamaño", "2.2.3 Defectos del esmalte"],
            "2.3. Higiene oral mecánica y química en el hogar": ["2.3.1 Técnicas de cepillado", "2.3.2 Colutorios"], 
            "2.4. Caries dental en el niño y el adolescente": ["2.4.1 Evaluación de riesgo cariogénico", "2.4.2 Flúor"],
            "2.5. Selladores de fosas y fisuras y uso de fluoruros": ["2.5.1 Indicaciones y técnicas"], 
            "2.6. Alteraciones pulpares en dientes deciduos y control del dolor": ["2.6.1 Terapia pulpar en dientes temporales", "2.6.2 Anestesia local en niños"],
            "2.7. Traumatismos de los dientes y tejidos de sostén": ["2.7.1 Clasificación y tratamiento"]
        },
        "3. Cirugía": {
            "3.1. Diagnóstico clínico y complementario": ["3.1.1 Exámenes de imagen", "3.1.2 Exámenes de laboratorio"], 
            "3.2. Anestesia": ["3.2.1 Técnicas anestésicas", "3.2.2 Complicaciones de la anestesia"], 
            "3.3. Principios de la técnica quirúrgica y exodoncia": ["3.3.1 Tiempos quirúrgicos", "3.3.2 Exodoncia simple y compleja"],
            "3.4. Indicaciones para cirugía pre protésica": ["3.4.1 Regularización de reborde", "3.4.2 Frenilectomía"], 
            "3.5. Infecciones bucales y maxilares": ["3.5.1 Abscesos y celulitis", "3.5.2 Osteomielitis"], 
            "3.6. Manejo de urgencias en cirugía": ["3.6.1 Hemorragias", "3.6.2 Alveolitis"]
        },
        "4. Rehabilitación Oral": {
            "4.1. Oclusión": ["4.1.1 Anatomía y fisiología de la ATM", "4.1.2 Tipos de oclusión"], 
            "4.2. Prótesis fija": ["4.2.1 Principios biomecánicos", "4.2.2 Preparación dentaria", "4.2.3 Cementación"], 
            "4.3. Prótesis parcial removible": ["4.3.1 Clasificación de Kennedy", "4.3.2 Diseño y componentes"], 
            "4.4 Prótesis total": ["4.4.1 Retención y estabilidad", "4.4.2 Impresiones"], 
            "4.5. Rehabilitación de dientes endodonciados": ["4.5.1 Postes y núcleos"]
        },
        "5. Endodoncia": {
            "5.1. Diagnóstico de alteraciones pulpares y periapicales": ["5.1.1 Clasificación de patología pulpar", "5.1.2 Clasificación de patología periapical"], 
            "5.2. Tratamiento endodóncico": ["5.2.1 Preparación biomecánica", "5.2.2 Obturación de conductos"],
            "5.3. Retratamiento y cirugía periapical": ["5.3.1 Indicaciones de retratamiento"], 
            "5.4. Complicaciones en endodoncia": ["5.4.1 Accidentes durante el tratamiento"]
        },
        "6. Periodoncia": {
            "6.1. Anatomía periodontal": ["6.1.1 Periodonto de inserción y protección"], 
            "6.2. Exámenes diagnósticos en periodoncia": ["6.2.1 Periodontograma", "6.2.2 Índices periodontales"],
            "6.3. Etiopatogenia de la enfermedad periodontal": ["6.3.1 Biofilm y cálculo dental"], 
            "6.4. Diagnóstico y clasificación de patologías periodontales (clasificación 2017)": ["6.4.1 Salud periodontal", "6.4.2 Gingivitis", "6.4.3 Periodontitis"],
            "6.5. Tratamiento periodontal": ["6.5.1 Fase higiénica", "6.5.2 Raspado y alisado radicular"]
        },
        "7. Patología Bucal": {
            "7.1. Lesiones de tejidos duros y diagnóstico diferencial": ["7.1.1 Quistes odontogénicos", "7.1.2 Tumores odontogénicos"], 
            "7.2. Patología de tejidos blandos y diagnóstico diferencial": ["7.2.1 Lesiones blancas y rojas", "7.2.2 Cáncer oral"],
            "7.3 Síndromes sistémicos": ["7.3.1 Manifestaciones orales de enfermedades sistémicas"]
        },
        "8. Farmacología": {
            "8.1. Anestésicos locales": ["8.1.1 Tipos de anestésicos", "8.1.2 Dosis máximas", "8.1.3 Vasoconstrictores"], 
            "8.2. Analgésicos y antiinflamatorios": ["8.2.1 Mecanismo de acción", "8.2.2 Dosis e indicaciones", "8.2.3 Interacciones"], 
            "8.3. Antibacterianos": ["8.3.1 Familias de antibióticos", "8.3.2 Profilaxis antibiótica", "8.3.3 Resistencia bacteriana"], 
            "8.4. Antivirales y antimicóticos": ["8.4.1 Indicaciones y dosis"]
        },
        "9. Medicina Interna": {
            "9.1. Enfermedades metabólicas": ["9.1.1 Diabetes Mellitus: manejo odontológico", "9.1.2 Hipotiroidismo e hipertiroidismo", "9.1.3 Osteoporosis"], 
            "9.2. Enfermedades respiratorias": ["9.2.1 Asma", "9.2.2 EPOC"], 
            "9.3. Embarazo": ["9.3.1 Manejo odontológico de la paciente embarazada"],
            "9.4. Urgencias y emergencias en Odontología": ["9.4.1 Crisis asmática", "9.4.2 Shock anafiláctico", "9.4.3 Broncoaspiración", "9.4.4 Crisis convulsiva", "9.4.5 Síncope"], 
            "9.5. Enfermedades cardiovasculares": ["9.5.1 Endocarditis bacteriana", "9.5.2 Hipertensión arterial", "9.5.3 Fiebre reumática"],
            "9.6. Trastornos de la coagulación": ["9.6.1 Manejo del paciente anticoagulado"], 
            "9.7. Soporte vital básico": ["9.7.1 Protocolos de RCP"]
        }
    }
}

# --- GESTIÓN DE BIBLIOTECA DE LIBROS ---
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
    """Lee libros subidos para obtener contexto en la respuesta"""
    texto_total = ""
    archivos = listar_archivos(carrera)
    lista_fuentes = []
    ruta_carpeta = os.path.join(DIRECTORIO_BASE, carrera)
    
    for nombre_archivo in archivos:
        try:
            ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
            reader = pypdf.PdfReader(ruta_completa)
            texto_archivo = f"\n--- INICIO FUENTE BIBLIOGRÁFICA: {nombre_archivo} ---\n"
            
            # Leemos primeras 60 paginas para contexto
            paginas_a_leer = min(len(reader.pages), 60)
            for i in range(paginas_a_leer):
                contenido = reader.pages[i].extract_text()
                if contenido:
                    texto_archivo += contenido + "\n"
            
            texto_archivo += f"\n--- FIN FUENTE: {nombre_archivo} ---\n"
            texto_total += texto_archivo
            lista_fuentes.append(nombre_archivo)
        except Exception as e:
            print(f"Error leyendo {nombre_archivo}: {e}")
            
    return texto_total, lista_fuentes

def extraer_paginas_pdf(archivo_pdf):
    """Extrae las páginas del PDF como una lista de textos"""
    paginas = []
    try:
        reader = pypdf.PdfReader(archivo_pdf)
        for page in reader.pages:
            texto = page.extract_text()
            if texto:
                paginas.append(texto)
    except Exception as e:
        st.error(f"Error al leer PDF de preguntas: {str(e)}")
    return paginas

# --- FUNCIONES DE IA ---

def configurar_api():
    with st.sidebar:
        st.header("⚙️ Configuración")
        api_key = st.text_input("Ingresa tu API Key de Google Gemini", type="password")
        st.divider()
        st.write("📚 **Biblioteca de Consulta**")
        st.caption("Sube aquí libros/guías para que la IA los lea y responda mejor.")
        inicializar_carpetas()
        for carrera in ESQUEMA_ACADEMICO.keys():
            n = len(listar_archivos(carrera))
            st.caption(f"- {carrera}: {n} libros")
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
    
    # 1. Leer libros subidos para contexto de respuesta
    texto_bibliografia, fuentes = leer_biblioteca_carrera(carrera_seleccionada)
    
    model, error = autodetectar_modelo(api_key)
    if error: return f"Error IA: {error}"
    
    contexto_extra = ""
    if texto_bibliografia:
        # Limitamos el contexto para evitar errores de payload excesivo
        contexto_extra = f"""
        FUENTES DE CONSULTA (Prioridad Alta):
        Usa esta información de los libros subidos ({', '.join(fuentes)}) para responder:
        
        {texto_bibliografia[:200000]} 
        """
    
    # Prompt optimizado
    prompt = f"""
    Actúa como un Evaluador Académico CACES (Ecuador).
    
    {contexto_extra}
    
    TAREA:
    Analiza el texto de entrada. Detecta preguntas, estandariza su formato y clasifícalas.
    
    REGLAS ESTRICTAS DE FORMATO:
    1. **Opciones**: 4 opciones, separadas por "|".
    2. **Respuesta Correcta**: COPIA EXACTA e IDÉNTICA de la opción correcta.
    3. **Feedback**: Separa CADA sección con DOBLE SALTO DE LÍNEA (\\n\\n):
       - Respuesta correcta: [Explicación]\\n\\n
       - Respuestas incorrectas: [Explicación]\\n\\n
       - Mnemotecnia/Tip: [Opcional]\\n\\n
       - Bibliografía: [Cita en formato VANCOUVER]

    REGLAS DE CLASIFICACIÓN (OBLIGATORIO):
    Usa SOLO el siguiente esquema. NO inventes temas.
    
    ESQUEMA OFICIAL ({carrera_seleccionada}):
    {json.dumps(ESQUEMA_ACADEMICO[carrera_seleccionada], ensure_ascii=False)}

    SALIDA JSON (Array):
    [
        {{
            "Pregunta": "Texto corregido...",
            "Opciones de Respuesta": "A|B|C|D",
            "Respuesta correcta": "C",
            "feedback": "...",
            "Carrera": "{carrera_seleccionada}",
            "Componente": "...",
            "Subcomponente": "...",
            "Tema": "..."
        }}
    ]
    
    TEXTO A PROCESAR: 
    {texto}
    """
    
    try:
        # Configuración de reintentos para evitar errores 503/504 esporádicos
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error, initial=10, multiplier=1.5, timeout=300)}
        response = model.generate_content(prompt, request_options=retry_policy)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        return f"Error procesando fragmento: {str(e)}"

def convertir_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Banco_Preguntas')
        worksheet = writer.sheets['Banco_Preguntas']
        workbook = writer.book
        format_wrap = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        
        for i, col in enumerate(df.columns):
            width = 50 if col == "feedback" else 20
            width = 40 if col == "Pregunta" else width
            worksheet.set_column(i, i, width, format_wrap if col == "feedback" else None)
            
    return output.getvalue()

# --- INTERFAZ UI ---

inicializar_carpetas()
api_key = configurar_api()

st.title("🎓 Gestor Académico Inteligente")
st.caption("Clasificación Estricta CACES + IA")

modo = st.radio("Selecciona una opción:", ["📝 Procesar Preguntas", "📚 Cargar Libros de Consulta"], horizontal=True)

if modo == "📚 Cargar Libros de Consulta":
    st.header("Biblioteca de Referencia")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Subir Libros/Guías")
        st.info("Sube aquí libros para ayudar a la IA a responder (NO afecta la estructura).")
        carrera_upload = st.selectbox("Carrera", list(ESQUEMA_ACADEMICO.keys()))
        archivo_pdf = st.file_uploader("Sube el PDF", type=["pdf"])
        
        if archivo_pdf and st.button("Guardar en Biblioteca", type="primary"):
            ruta = guardar_pdf(archivo_pdf, carrera_upload)
            st.success(f"✅ Libro guardado para: {carrera_upload}")
            st.balloons()
            time.sleep(1)
            st.rerun()

    with col2:
        st.subheader("Libros Disponibles")
        for carrera in ESQUEMA_ACADEMICO.keys():
            archivos = listar_archivos(carrera)
            with st.expander(f"📂 {carrera} ({len(archivos)})"):
                if archivos:
                    for f in archivos:
                        st.markdown(f"📖 `{f}`")
                else:
                    st.caption("Sin libros")

elif modo == "📝 Procesar Preguntas":
    st.header("Procesamiento de Exámenes")
    
    col_conf, col_work = st.columns([1, 2])
    with col_conf:
        st.info("Configuración")
        carrera_proceso = st.selectbox("Selecciona Carrera", list(ESQUEMA_ACADEMICO.keys()))
        archivos_disp = listar_archivos(carrera_proceso)
        if archivos_disp:
            st.success(f"✅ {len(archivos_disp)} libros de consulta disponibles.")
        else:
            st.info("ℹ️ No hay libros subidos. La IA usará su conocimiento general.")

    with col_work:
        tab_txt, tab_xls, tab_pdf = st.tabs(["Pegar Texto", "Subir Excel", "Subir PDF de Preguntas"])
        texto_final = None
        origen_datos = None
        
        with tab_txt:
            txt = st.text_area("Pega las preguntas aquí:", height=200)
            if st.button("Procesar Texto"): 
                texto_final = txt
                origen_datos = "texto"
        
        with tab_xls:
            f = st.file_uploader("Sube Excel", type=["xlsx"])
            if f:
                df = pd.read_excel(f)
                c = st.selectbox("Columna Pregunta", df.columns)
                if st.button("Procesar Excel"):
                    texto_final = "\n---\n".join(df[c].astype(str).tolist())
                    origen_datos = "excel"
        
        with tab_pdf:
            pdf_q = st.file_uploader("Sube PDF con preguntas", type=["pdf"])
            if pdf_q and st.button("Procesar PDF de Preguntas"):
                origen_datos = "pdf"
                # Aquí no extraemos todo el texto de golpe, lo manejamos abajo por lotes

    # Lógica de Procesamiento
    if origen_datos == "pdf" and pdf_q:
        with st.status("🚀 Iniciando procesamiento por lotes...", expanded=True) as status:
            paginas = extraer_paginas_pdf(pdf_q)
            if not paginas:
                st.error("No se pudo leer el PDF.")
                st.stop()
            
            # --- ESTRATEGIA DE BATCHING (LOTES) ---
            # Procesamos de 2 en 2 páginas para evitar errores de Timeout (Error 504)
            TAMANO_LOTE = 2 
            lotes = ["\n".join(paginas[i:i+TAMANO_LOTE]) for i in range(0, len(paginas), TAMANO_LOTE)]
            
            resultados_totales = []
            barra_progreso = st.progress(0)
            
            st.write(f"📄 Documento dividido en {len(lotes)} partes para análisis detallado.")
            
            for i, lote_texto in enumerate(lotes):
                st.write(f"Analizando parte {i+1} de {len(lotes)}...")
                res_parcial = procesar_con_ia(lote_texto, api_key, carrera_proceso)
                
                if isinstance(res_parcial, list):
                    resultados_totales.extend(res_parcial)
                else:
                    st.warning(f"⚠️ Hubo un problema en la parte {i+1}: {res_parcial}")
                
                barra_progreso.progress((i + 1) / len(lotes))
                time.sleep(1) # Pequeña pausa para no saturar la API
            
            if resultados_totales:
                status.update(label="¡Análisis Completo!", state="complete", expanded=False)
                df_res = pd.DataFrame(resultados_totales)
                
                st.divider()
                st.subheader("✅ Resultados Consolidados")
                editado = st.data_editor(
                    df_res, 
                    num_rows="dynamic", 
                    use_container_width=True,
                    column_config={"feedback": st.column_config.TextColumn("Feedback", width="large")}
                )
                
                st.download_button(
                    "📥 Descargar Excel Final", 
                    convertir_excel(editado), 
                    "banco_preguntas_caces.xlsx", 
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                    type="primary"
                )
            else:
                st.error("No se pudieron extraer preguntas válidas del PDF.")

    elif texto_final:
        # Procesamiento normal para texto corto o Excel
        with st.status("🧠 Analizando...", expanded=True) as status:
            res = procesar_con_ia(texto_final, api_key, carrera_proceso)
            
            if isinstance(res, list):
                status.update(label="¡Proceso Completado!", state="complete", expanded=False)
                df_res = pd.DataFrame(res)
                
                st.divider()
                st.subheader("Resultados")
                editado = st.data_editor(
                    df_res, 
                    num_rows="dynamic", 
                    use_container_width=True,
                    column_config={"feedback": st.column_config.TextColumn("Feedback", width="large")}
                )
                
                st.download_button(
                    "📥 Descargar Excel", 
                    convertir_excel(editado), 
                    "banco_preguntas_caces.xlsx", 
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                    type="primary"
                )
            else:
                status.update(label="Error", state="error")
                st.error("Error en el procesamiento:")
                st.warning(res)
