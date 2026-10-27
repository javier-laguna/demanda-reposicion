# Cómo se llevaría a producción

El repo corre completo en una laptop o en Docker, con datos locales, MLflow en SQLite y secretos en `.env`. Este
documento describe lo que cambiaría para operarlo en una nube. Ninguna de estas piezas se usa aquí: es el diseño, no
una implementación. Los ejemplos usan GCP, pero el mismo esquema aplica en AWS o Azure.

## Secretos

| Hoy | En producción |
|---|---|
| `.env` local, gitignored, leído como `SecretStr` | Secret Manager, con un secreto por credencial y versionado |
| La key del LLM en la laptop | La key vive solo en el gateway LLM corporativo; el pipeline no la ve |
| gitleaks en pre-commit y en CI | Igual, más la detección de secretos de la plataforma en el repo |

- **Identidad sin llaves:**
  - el job corre con una cuenta de servicio propia, por Workload Identity (en GKE) o la identidad del servicio (en
    Cloud Run), y lee sus secretos sin descargar llaves JSON;
  - CI se autentica contra GCP con OIDC (Workload Identity Federation de GitHub Actions), sin secretos de larga
    vida en GitHub.
- **Mínimo privilegio:**
  - la cuenta del pipeline lee el bucket de datos crudos y escribe solo en el de artefactos;
  - la del tablero solo lee artefactos;
  - la del servidor MCP solo lee artefactos y no tiene salida a internet.
- **Rotación:**
  - las keys de proveedores LLM rotan en el gateway sin tocar el pipeline;
  - Secret Manager guarda versiones, y el job toma siempre la última habilitada.
- **LLM por gateway:**
  - `LLM_BASE_URL` ya permite apuntar a un gateway compatible con OpenAI;
  - el gateway centraliza la key, el registro de uso, los límites de gasto por equipo y la política de qué datos
    pueden salir. El freno de gasto del código queda como segunda línea.

## Datos

- **Almacenamiento:** los CSV de entrada van a un bucket de GCS con versionado de objetos y retención. El manifiesto
  de hashes (`data/manifest.sha256`) se conserva: cada corrida verifica que lee la versión esperada y la registra en
  sus artefactos, como ya hace con `fuente.sha256`.
- **Versionado de datos:** DVC es una opción si el equipo quiere ligar cada commit a una versión de datos. Con el
  versionado de GCS más el manifiesto, la trazabilidad mínima ya existe.
- **Contratos:** los contratos de Pandera corren en la ingesta, antes de cualquier etapa. Un contrato roto detiene la
  corrida y alerta; los problemas conocidos se miden en el reporte de calidad, como hoy.
- **Datos personales:** el dataset es agregado por tienda × categoría × día. Si en producción llegaran datos a nivel
  ticket o cliente, la agregación se haría antes de que salgan del perímetro de datos, y los agentes seguirían viendo
  solo agregados.

## Escala

El problema de la prueba son 480 series. Un retailer real tiene decenas de miles de tiendas × SKU.

- **Features y agregación:** el pipeline usa Polars sobre un solo nodo. A nivel SKU, las mismas transformaciones
  (rezagos con `shift` antes de `rolling`, calendario, atributos de tienda) se escriben en PySpark sobre Dataproc o
  en SQL de BigQuery. Las dos particionan por serie, y la ventana de cada serie no cruza particiones.
- **Modelo global:** LightGBM ya es un modelo global con features por serie, el patrón que escala. A nivel SKU se
  entrena por grupo (categoría o región) o con entrenamiento distribuido. Chronos-2 sin entrenar escala por
  inferencia en lote y se beneficiaría de GPU.
- **Backtest:** los cortes son independientes entre sí y se paralelizan como tareas separadas.

## Operación

- **Cadencia:** una corrida semanal agendada (Cloud Scheduler más Cloud Run Jobs, o un DAG de Airflow), con las
  mismas etapas de `demanda pipeline`. Cada etapa escribe sus artefactos con el hash de sus fuentes, así que una
  corrida a medias se retoma con `--desde`.
- **Registro de modelos:**
  - MLflow pasa de SQLite a un servidor con base de datos administrada y artefactos en GCS;
  - el campeón se promueve en el registro solo cuando la regla de campeón lo decide, con el JSON de selección como
    evidencia adjunta.
- **Monitor de drift:**
  - corre después de cada corrida semanal;
  - las alertas inesperadas van al canal del equipo, con su recomendación (`revisar_captura`, `plan_de_evento`,
    `reentrenar`, `reevaluar_campeon`);
  - antes de automatizar el reentrenamiento hay que corregir la limitación conocida del chequeo de nivel: con un
    segundo año de datos, la referencia pasa a ser la misma semana del año anterior.
- **Capa de agentes:**
  - `make insights` corre después del monitor y deja `insights.json` para el tablero;
  - el servidor MCP se despliega detrás de autenticación (IAP o el proxy de identidad de la plataforma), solo dentro
    de la red interna y en modo de solo lectura;
  - los clientes externos (otros agentes del negocio) consultan por HTTP.
- **Observabilidad:**
  - cada etapa registra duración y conteos;
  - el gasto del LLM se reporta desde el gateway;
  - las cifras rechazadas por el validador y las caídas a plantilla se vigilan como métricas de calidad de la capa
    de agentes.
