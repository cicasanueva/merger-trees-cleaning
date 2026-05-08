
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Catalina Casanueva, 21/08/2025, cicasanueva@uc.cl
# ============================================================
# Pipeline unificado: STEP 4 + STEP 5 con export corregido
# ============================================================
import traceback
from utils import * 
import os, sys, json, re, ast, csv
import h5py, numpy as np, pandas as pd, networkx as nx
from tqdm import tqdm
from collections import defaultdict
from ast import literal_eval

def safe_json_list(val):
    """Convierte cadenas/JSON a listas de Python con tolerancia a formatos. Se usa para deserializar columnas de IDs."""
    """
    Convierte cualquier iterable (incluyendo arrays NumPy) a lista de int nativos,
    y luego a string JSON serializable.
    """
    try:
        return json.dumps([int(x) for x in val])
    except Exception as e:
        print(f"⚠️ Error convirtiendo a JSON: {e}")
        return "[]"



def depura_mergers_trees(sim_name, target_subfind_id, snap_target):
    """Paso 4 (depuración): detecta fusiones post-infall y computa sus ratios; mantiene la lógica original."""
    # ==== CONFIGURACIÓN GENERAL ====
    #sim_name = "147062"
    #target_subfind_id = 2627
    #snap_target = 128
    uid_target = f"{snap_target}{target_subfind_id:06d}"
    carpeta_out = "Datos/Baryon_Mergers_Depurados/"
    # ==== PATHS DE ENTRADA ====
    path_hdf5 = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"

    os.makedirs(carpeta_out, exist_ok=True)
    
    # ========================================
    # DEFINICIÓN DE FUNCIONES PARA CLASIFICACIÓN
    # ========================================

    def low_transfer(row):
        """
        Evalúa si todas las fracciones de transferencia relevantes
        están por debajo del umbral definido (frac_thresh).
        """
        return all(
            (row[col] < frac_thresh)
            for col in frac_cols
            if not pd.isna(row[col])
        )

    def clasificar_grupo(row):
        """
        Clasifica un merger en uno de los seis grupos (A1, A2, B1, B2, C1, C2)
        según su historial orbital y la eficiencia de transferencia de masa.
        """
        if row["flag_always_inside_r200"]:
            return "A1" if low_transfer(row) else "A2"
        elif row["flag_never_inside_r200"]:
            return "B1" if low_transfer(row) else "B2"
        else:
            return "C1" if low_transfer(row) else "C2"

    def get_xlabel(dfg):
        """
        Devuelve la etiqueta del eje X apropiada para los gráficos,
        dependiendo del tipo de merger (entrada, salida o cruce del halo).
        """
        if dfg["flag_always_inside_r200"].all():
            return "Mass at first snapshot [$M_\\odot$]"
        elif dfg["flag_never_inside_r200"].all():
            return "Mass at last snapshot [$M_\\odot$]"
        else:
            return "Mass at $t_\\mathrm{infall}$ [$M_\\odot$]"

    def get_snap_plot(row):
        """
        Determina el snapshot de referencia para gráficos.
        - Si el merger está siempre o nunca en el halo: usar snap de uid_infall.
        - Si entra al halo (tipo C): usar snap_infall directamente.
        """
        if row["flag_always_inside_r200"] or row["flag_never_inside_r200"]:
            return split_unique_id(int(row["uid_infall"]))[0]
        else:
            return row["snap_infall"]

    def limpiar_main_branch(uids):
        """
        Recibe una lista de uids que representan una rama principal
        y devuelve un diccionario limpio con un único uid por snapshot.
        """
        uids = list(map(int, uids))
        dict_por_snap = {
            uid // 10**6: uid for uid in uids
        }
        return {snap: uid for snap, uid in sorted(dict_por_snap.items())}

    # ========================================
    # CARGA DEL ÁRBOL DE MERGERS
    # ========================================

    tree_1 = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)  # árbol hacia atrás
    tree_2 = nx.read_multiline_adjlist(path_tree)  # árbol hacia adelante

    def main_branch_completa(uid):
        """
        Devuelve la rama principal completa de un merger (hacia atrás y adelante).
        """
        subtree_fwd = nx.dfs_tree(tree_1, uid)
        rama_fwd = get_main_branch_unique_ids(subtree_fwd, uid)
        subtree_bwd = nx.dfs_tree(tree_2, uid)
        rama_bwd = get_main_branch_unique_ids(subtree_bwd, uid)[1:][::-1]
        return rama_bwd + rama_fwd

    def contar_particulas(uid, componente):
        """
        Devuelve el número de partículas de una componente ('stars', 'gas', 'dm')
        asociadas a un merger dado. Si no hay datos disponibles, retorna None.
        """
        try:
            return len(dict_particulas_infall[uid][componente])
        except KeyError:
            return None

    # ========================================
    # CARGA Y CLASIFICACIÓN DE MERGERS BARIÓNICOS
    # ========================================

    # ----------------------------------------
    # 1. Cargar el archivo CSV con mergers detectados en la galaxia central 
    # ----------------------------------------
    df = pd.read_csv(f"Datos/Baryon_Mergers/baryon_mergers_{sim_name}_gx_{target_subfind_id}_summary.csv")

    # ----------------------------------------
    # 2. Filtrar mergers con al menos una componente bariónica (estrellas o gas)
    # ----------------------------------------
    df = df[(df["mass_stars_infall"] > 0) | (df["mass_gas_infall"] > 0)]
    
    # >>>>>> GUARD <<<<<<
    if df.empty:
        # Esquema mínimo que el resto del pipeline espera
        columnas_minimas = [
            "uid_infall","uid_disruption","grupo",
            "mass_stars_infall","mass_gas_infall","mass_dm_infall",
            # columnas que los filtros podrían leer (se crearán vacías)
            "flag_overlap_eliminar_inicioC2","flag_overlap_eliminar_contieneC2","flag_overlap_eliminar_sinC",
            "uids_asociadas_overlap_inicioC2","uids_asociadas_overlap_contieneC2","uids_asociadas_overlap_sinC",
            "uid_disruption_overlap_inicioC2","uid_disruption_overlap_contieneC2","uid_disruption_overlap_sinC",
            "uid_overlap_conservado_inicioC2","uid_overlap_conservado_contieneC2","uid_overlap_conservado_sinC"
        ]
        df_vacio = pd.DataFrame(columns=columnas_minimas)

        # ruta/archivo de salida (ajústalo si tu export final usa otro nombre)
        carpeta_out = "Datos/Baryon_Mergers_Depurados/"
        os.makedirs(carpeta_out, exist_ok=True)
        out_csv = os.path.join(carpeta_out, f"baryon_mergers_dep_{sim_name}_gx_{target_subfind_id}.csv")

        df_vacio.to_csv(out_csv, index=False)
        print(f"[OK] 0 mergers bariónicos: exportado vacío con esquema estándar → {out_csv}")
        return  # terminamos limpio sin continuar filtros


    # ----------------------------------------
    # 3. Definir umbral de baja transferencia de masa
    # ----------------------------------------
    frac_thresh = 1e-1  # Umbral: 10%

    # ----------------------------------------
    # 4. Columnas que contienen fracciones de masa transferida
    # ----------------------------------------
    frac_cols = [
        "Mstell_frac_t_infall_in_central_z0",       # estrellas totales transferidas
        "Mgas_frac_t_infall_in_central_z0",         # gas transferido
        "Mdm_frac_t_infall_in_central_z0",          # DM transferida
        "frac_star1_from_gas_infall_in_central_z0", # estrellas 1ª generación
        "frac_star2_from_gas_infall_in_central_z0", # estrellas 2ª generación
    ]

    # ----------------------------------------
    # 5. Clasificar cada merger en su grupo correspondiente
    # ----------------------------------------
    df["grupo"] = df.apply(clasificar_grupo, axis=1)

    # ----------------------------------------
    # 6. Descripciones textuales para cada grupo
    # ----------------------------------------
    descripciones = {
        "A1": "Siempre dentro del halo central, con bajo traspaso",
        "A2": "Siempre dentro del halo central, con alto traspaso",
        "B1": "Nunca dentro del halo central, con bajo traspaso",
        "B2": "Nunca dentro del halo central, con alto traspaso",
        "C1": "Entraron al halo en algún momento, con bajo traspaso",
        "C2": "Entraron al halo en algún momento, con alto traspaso",
    }

    # ----------------------------------------
    # 7. Imprimir resumen de estadísticas por grupo
    # ----------------------------------------
    print("=== Cuantificación por grupo (solo mergers bariónicos) ===\n")

    for g in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        dfg = df[df["grupo"] == g]
        n = len(dfg)
        desc = descripciones[g]
        print(f"Grupo {g}: {desc}")
        print(f"   - N mergers             : {n}")

        if n > 0:
            mmin = dfg["mass_stars_infall"].min()
            mmax = dfg["mass_stars_infall"].max()
            print(f"   - M⋆ infall [min, max] : [{mmin:.2e}, {mmax:.2e}] M☉")

            n_solo_gas   = len(dfg[(dfg["mass_stars_infall"] == 0) & (dfg["mass_gas_infall"] > 0)])
            n_solo_stars = len(dfg[(dfg["mass_stars_infall"] > 0) & (dfg["mass_gas_infall"] == 0)])
            n_ambos      = len(dfg[(dfg["mass_stars_infall"] > 0) & (dfg["mass_gas_infall"] > 0)])

            print(f"   - Solo gas             : {n_solo_gas}")
            print(f"   - Solo estrellas       : {n_solo_stars}")
            print(f"   - Gas + estrellas      : {n_ambos}\n")
        else:
            print(f"   - M⋆ infall             : N/A\n")


    # ========================================
    # CARGA DE PARTÍCULAS EN t_infall PARA TODOS LOS MERGERS
    # ========================================

    # ----------------------------------------
    # 1. Inicializar diccionario para guardar partículas por merger
    # ----------------------------------------
    dict_particulas_infall = {}

    # ----------------------------------------
    # 2. Leer partículas de gas, estrellas y DM asociadas a cada subhalo en t_infall
    #    Este diccionario será clave para comparar componentes compartidas entre mergers.
    # ----------------------------------------
    with h5py.File(path_hdf5, "r") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Cargando partículas a t_infall (todos los grupos)"):
            uid = int(row["uid_infall"])                   # ID único del subhalo en t_infall
            grupo = row["grupo"]                           # Grupo al que pertenece el merger
            snap, subf = split_unique_id(uid)              # Snapshot y subfind
            gname = f"SnapNumber_{snap}"                   # Ruta al snapshot en HDF5

            # ----------- Gas (PartType0) -----------
            ids_gas = f[gname]["PartType0/ParticleIDs"][:]            # IDs de todas las partículas de gas
            subf_gas = f[gname]["PartType0/SubFindNumber"][:]         # Subhalo al que pertenece cada partícula
            ids_gas_subf = ids_gas[subf_gas == subf]                  # Selección de partículas del subhalo

            # ----------- Estrellas (PartType4) -----------
            ids_star = f[gname]["PartType4/ParticleIDs"][:]           # IDs de todas las estrellas
            subf_star = f[gname]["PartType4/SubFindNumber"][:]        # Subhalo asociado
            ids_star_subf = ids_star[subf_star == subf]               # Estrellas de este merger

            # ----------- Materia Oscura (PartType1) -----------
            ids_dm = f[gname]["PartType1/ParticleIDs"][:]             # IDs de todas las partículas DM
            subf_dm = f[gname]["PartType1/SubFindNumber"][:]          # Subhalo de cada una
            ids_dm_subf = ids_dm[subf_dm == subf]                     # DM de este subhalo

            # ----------- Guardar resultados en diccionario -----------
            clave = f"{uid}"
            dict_particulas_infall[clave] = {
                "snap": snap,
                "gas": ids_gas_subf,
                "stars": ids_star_subf,
                "dm": ids_dm_subf,
                "grupo": grupo,
            }


    # ========================================
    # DETECCIÓN DE OVERLAPS FUERTES POR COMPONENTE DOMINANTE
    # ========================================
    # Este bloque identifica mergers que comparten ≥90% de su componente dominante
    # (según masa al infall) con otra estructura. Si además poseen otras componentes 
    # con más de 100 partículas, se exige que dichas componentes estén compartidas 
    # en ≥50% con la misma estructura.
    #
    # Esta lógica permite detectar duplicaciones físicas claras, 
    # y a la vez mantener mergers que podrían representar estructuras distintas.


    dict_overlaps = {}

    for uid_a in tqdm(dict_particulas_infall, desc="🔍 Buscando overlaps fuertes"):
        info_a = dict_particulas_infall[uid_a]

        # --- Clasificación bariónica del merger A ---
        has_gas = len(info_a["gas"]) > 0
        has_stars = len(info_a["stars"]) > 0
        if has_gas and has_stars:
            tipo = "gas and stars"
        elif has_gas:
            tipo = "only gas"
        elif has_stars:
            tipo = "only stars"
        else:
            tipo = "no baryons"

        grupo_a = info_a["grupo"]

        # --- Obtener snapshot y masas desde el DataFrame ---
        row_df = df[df["uid_infall"] == int(uid_a)]
        if row_df.empty:
            continue  # ignorar si no está en df, por consistencia

        snap, _ = split_unique_id(int(uid_a))
        mstar = row_df["mass_stars_infall"].values[0]
        mgas  = row_df["mass_gas_infall"].values[0]
        mdm   = row_df["mass_dm_infall"].values[0]

        # --- Determinar componente más masiva ---
        masas = {"stars": mstar, "gas": mgas, "dm": mdm}
        comp_dominante = max(masas, key=masas.get)

        # --- Obtener sets de partículas para A ---
        ids_a_all = {comp: set(info_a[comp]) for comp in ["gas", "stars", "dm"]}

        # --- Inicializar lista de overlaps válidos con otros mergers B ---
        overlaps_validos = []

        for uid_b, info_b in dict_particulas_infall.items():
            if uid_a == uid_b:
                continue  # no comparar consigo mismo

            ids_b_all = {comp: set(info_b[comp]) for comp in ["gas", "stars", "dm"]}

            # -- Paso 1: verificar overlap fuerte en la componente dominante --
            ids_a_dom = ids_a_all[comp_dominante]
            ids_b_dom = ids_b_all[comp_dominante]
            if not ids_a_dom or not ids_b_dom:
                continue

            shared_dom = ids_a_dom & ids_b_dom
            frac_a_to_b_dom = 100 * len(shared_dom) / len(ids_a_dom)

            if frac_a_to_b_dom < 90:
                continue  # no hay overlap fuerte dominante, descartamos B

            # -- Paso 2: verificar consistencia en otras componentes si son relevantes --
            otras_componentes = [
                comp for comp in ["gas", "stars", "dm"]
                if comp != comp_dominante and len(ids_a_all[comp]) > 100
            ]

            cumple_otros = True
            for comp in otras_componentes:
                ids_a = ids_a_all[comp]
                ids_b = ids_b_all[comp]
                if not ids_b:
                    cumple_otros = False
                    break
                shared = ids_a & ids_b
                frac_a_to_b = 100 * len(shared) / len(ids_a)
                if frac_a_to_b < 50:
                    cumple_otros = False
                    break

            if cumple_otros:
                # Guardar información de este B
                frac_b_to_a_dom = 100 * len(shared_dom) / len(ids_b_dom)
                grupo_b = info_b["grupo"]
                overlaps_validos.append((uid_b, frac_a_to_b_dom, frac_b_to_a_dom, grupo_b))

        # --- Guardar resultado si hubo al menos un overlap válido ---
        if overlaps_validos:
            dict_overlaps[uid_a] = {
                "snap": snap,
                "tipo": tipo,
                "grupo": grupo_a,
                "mass_stars": mstar,
                "mass_gas": mgas,
                "mass_dm": mdm,
                "overlaps": {
                    comp_dominante: sorted(overlaps_validos, key=lambda x: -x[1])  # ordenar por % descendente
                }
            }


    # Agrupar mergers que, según el merger tree, aparecen como estructuras distintas,
    # pero que comparten una fracción significativa de partículas bariónicas al t_infall.

    # ----------------------------------------
    # 1. Crear grafo vacío
    # ----------------------------------------
    G = nx.Graph()

    # ----------------------------------------
    # 2. Agregar nodos: todos los mergers con overlaps
    # ----------------------------------------
    for uid_a in dict_overlaps:
        G.add_node(uid_a)

    # ----------------------------------------
    # 3. Agregar aristas: si A comparte >90% de alguna componente con B, conectar A y B
    # ----------------------------------------
    for uid_a, info in dict_overlaps.items():
        for comp, overlaps in info["overlaps"].items():
            for uid_b, frac_a_to_b, frac_b_to_a, grupo_b in overlaps:
                G.add_edge(uid_a, uid_b, componente=comp, frac_a=frac_a_to_b, frac_b=frac_b_to_a)
    # ----------------------------------------
    # 4. Extraer componentes conectadas (grupos de mergers con overlaps mutuos)
    # ----------------------------------------
    componentes_overlap = list(nx.connected_components(G))

    # Ordenar cada grupo por snapshot para inspección más clara
    componentes_overlap = [sorted(list(grupo), key=lambda uid: int(uid)) for grupo in componentes_overlap]

    # Mostrar resultados
    print(f"Se encontraron {len(componentes_overlap)} grupos de mergers con overlaps fuertes:")
    for i, grupo in enumerate(componentes_overlap):
        print(f"Grupo {i+1} ({len(grupo)} mergers): {grupo}")

    uids_implicados = sorted(list(G.nodes), key=int)
    print("🔍 Total de mergers implicados en overlaps fuertes:", len(uids_implicados))


    # ============================================================
    # FILTRO 1: Grupos con overlap fuerte donde el primer merger es del grupo C2
    # ============================================================
    # Este filtro identifica duplicaciones dentro de grupos de mergers con alto solapamiento de partículas,
    # en los que el merger más temprano en el tiempo (menor snapshot) pertenece al grupo C2.

    # El criterio de conservación es el siguiente:
    # - Se conserva el merger C2 (llamado A).
    # - Se eliminan los demás mergers del grupo (llamados B) si:
    #   · Su componente dominante está contenida al ≥75% en A.
    #   · Y si alguna componente secundaria tiene más de 100 partículas, debe estar contenida al ≥50% en A.

    # ============================================================
    # Columnas agregadas al DataFrame por este filtro
    # ============================================================

    # → Flags y asociaciones:
    # - flag_overlap_eliminar_inicioC2: True para los mergers eliminados por solapamiento estructural con un C2 anterior.
    # - uid_overlap_conservado_inicioC2: UID del merger C2 que reemplaza al merger eliminado.
    # - uid_disruption_overlap_inicioC2: UID de disrupción más tardía entre los mergers eliminados, adoptado por el merger conservado.
    # - uids_asociadas_overlap_inicioC2: lista de mergers eliminados por este criterio.

    # → Fracciones de partículas compartidas:
    # Se calculan fracciones de solapamiento entre el merger conservado (A) y cada merger eliminado (B), para cada componente:

    # - frac_{comp}_A_en_Bs_overlap_inicioC2: % de partículas de A contenidas en cada B (donde comp = star, gas, dm).
    # - frac_{comp}_Bs_en_A_overlap_inicioC2: % de partículas de cada B contenidas en A.

    # Estas fracciones permiten validar la absorción estructural exigida por el filtro.


    # Inicializar columnas en el DataFrame
    df["flag_overlap_eliminar_inicioC2"] = False
    df["uid_disruption_overlap_inicioC2"] = None
    df["uids_asociadas_overlap_inicioC2"] = None
    df["uid_overlap_conservado_inicioC2"] = None

    for comp in ["star", "gas", "dm"]:
        df[f"frac_{comp}_A_en_Bs_overlap_inicioC2"] = None  # % de A contenido en cada B
        df[f"frac_{comp}_Bs_en_A_overlap_inicioC2"] = None  # % de cada B contenido en A

    # Aplicar filtro por grupo de overlap
    for grupo in componentes_overlap:
        grupo_ordenado = sorted(grupo, key=lambda uid: split_unique_id(int(uid))[0])
        uid_primero = int(grupo_ordenado[0])

        fila_primero = df[df["uid_infall"] == uid_primero]
        if fila_primero.empty or fila_primero["grupo"].values[0] != "C2":
            continue  # solo procesar si el primer merger es del grupo C2

        grupo_int = [int(uid) for uid in grupo_ordenado]

        # Obtener partículas del merger conservado (A)
        p_star_A = set(dict_particulas_infall[str(uid_primero)]["stars"])
        p_gas_A  = set(dict_particulas_infall[str(uid_primero)]["gas"])
        p_dm_A   = set(dict_particulas_infall[str(uid_primero)]["dm"])

        n_star_A, n_gas_A, n_dm_A = len(p_star_A), len(p_gas_A), len(p_dm_A)

        uids_asociadas, uids_disruption, snaps = [], [], []
        fracs_star_A_en_Bs, fracs_gas_A_en_Bs, fracs_dm_A_en_Bs = [], [], []
        fracs_star_Bs_en_A, fracs_gas_Bs_en_A, fracs_dm_Bs_en_A = [], [], []

        for uid in grupo_int:
            if uid == uid_primero:
                continue  # no compararse consigo mismo

            # Partículas del merger asociado (B)
            p_star_B = set(dict_particulas_infall[str(uid)]["stars"])
            p_gas_B  = set(dict_particulas_infall[str(uid)]["gas"])
            p_dm_B   = set(dict_particulas_infall[str(uid)]["dm"])

            n_star_B, n_gas_B, n_dm_B = len(p_star_B), len(p_gas_B), len(p_dm_B)

            # Obtener masas de B desde el DataFrame
            row_b = df[df["uid_infall"] == uid]
            if row_b.empty:
                continue

            mass_b = {
                "stars": row_b["mass_stars_infall"].values[0],
                "gas": row_b["mass_gas_infall"].values[0],
                "dm": row_b["mass_dm_infall"].values[0],
            }
            comp_dom_b = max(mass_b, key=mass_b.get)

            # === Fracciones ===
            # A en B
            frac_star_A_in_B = 100 * len(p_star_A & p_star_B) / n_star_A if n_star_A > 0 else None
            frac_gas_A_in_B  = 100 * len(p_gas_A  & p_gas_B)  / n_gas_A  if n_gas_A  > 0 else None
            frac_dm_A_in_B   = 100 * len(p_dm_A   & p_dm_B)   / n_dm_A   if n_dm_A   > 0 else None

            # B en A
            frac_star_B_in_A = 100 * len(p_star_A & p_star_B) / n_star_B if n_star_B > 0 else None
            frac_gas_B_in_A  = 100 * len(p_gas_A  & p_gas_B)  / n_gas_B  if n_gas_B  > 0 else None
            frac_dm_B_in_A   = 100 * len(p_dm_A   & p_dm_B)   / n_dm_B   if n_dm_B   > 0 else None

            fracs_star_A_en_Bs.append(frac_star_A_in_B)
            fracs_gas_A_en_Bs.append(frac_gas_A_in_B)
            fracs_dm_A_en_Bs.append(frac_dm_A_in_B)

            fracs_star_Bs_en_A.append(frac_star_B_in_A)
            fracs_gas_Bs_en_A.append(frac_gas_B_in_A)
            fracs_dm_Bs_en_A.append(frac_dm_B_in_A)

            # === Validación física: contenido estructural ===
            fracs_B_en_A = {
                "stars": frac_star_B_in_A,
                "gas": frac_gas_B_in_A,
                "dm": frac_dm_B_in_A,
            }

            # Dominante debe estar ≥75% en A
            if fracs_B_en_A[comp_dom_b] is None or fracs_B_en_A[comp_dom_b] < 75:
                continue

            # Secundarias con >100 partículas deben estar ≥50%
            cumple_secundarias = True
            for comp in ["stars", "gas", "dm"]:
                if comp == comp_dom_b:
                    continue
                n_part_B = len(dict_particulas_infall[str(uid)][comp])
                frac_in = fracs_B_en_A[comp]
                if n_part_B > 100 and (frac_in is None or frac_in < 50):
                    cumple_secundarias = False
                    break

            if not cumple_secundarias:
                continue

            # Marca B como eliminable
            df.loc[df["uid_infall"] == uid, "flag_overlap_eliminar_inicioC2"] = True
            df.loc[df["uid_infall"] == uid, "uid_overlap_conservado_inicioC2"] = uid_primero
            uids_asociadas.append(uid)

            # Guardar UID de disrupción
            uid_dis = row_b["uid_disruption"].values[0]
            uids_disruption.append(uid_dis)
            snaps.append(split_unique_id(int(uid_dis))[0])

        # Guardar info en el merger conservado (A)
        if uids_asociadas:
            idx_max = snaps.index(max(snaps))
            df.loc[df["uid_infall"] == uid_primero, "uid_disruption_overlap_inicioC2"] = uids_disruption[idx_max]
            df.loc[df["uid_infall"] == uid_primero, "uids_asociadas_overlap_inicioC2"] = str(uids_asociadas)

            df.loc[df["uid_infall"] == uid_primero, "frac_star_A_en_Bs_overlap_inicioC2"] = str(fracs_star_A_en_Bs)
            df.loc[df["uid_infall"] == uid_primero, "frac_gas_A_en_Bs_overlap_inicioC2"]  = str(fracs_gas_A_en_Bs)
            df.loc[df["uid_infall"] == uid_primero, "frac_dm_A_en_Bs_overlap_inicioC2"]   = str(fracs_dm_A_en_Bs)

            df.loc[df["uid_infall"] == uid_primero, "frac_star_Bs_en_A_overlap_inicioC2"] = str(fracs_star_Bs_en_A)
            df.loc[df["uid_infall"] == uid_primero, "frac_gas_Bs_en_A_overlap_inicioC2"]  = str(fracs_gas_Bs_en_A)
            df.loc[df["uid_infall"] == uid_primero, "frac_dm_Bs_en_A_overlap_inicioC2"]   = str(fracs_dm_Bs_en_A)


    # ============================================================
    # RESUMEN DE RESULTADOS: Filtro 1 - overlap_inicio_C2
    # ============================================================
    # Este bloque resume el impacto del primer filtro de limpieza por overlaps fuertes.
    # En este filtro, se conservaron mergers del grupo C2 (más antiguos) dentro de cada
    # componente conectada, y se eliminaron los mergers posteriores si estaban mayoritariamente
    # contenidos en dicho C2 según fracciones de partículas (estrellas, gas, DM).
    # Se reportan:
    # - Los mergers C2 conservados y los mergers que absorbieron.
    # - Las fracciones de partículas que justifican la absorción.
    # - Los mergers eliminados y el C2 correspondiente que los reemplaza.

    df_conservados_overlap_inicioC2 = df[
        (df["flag_overlap_eliminar_inicioC2"] == False) &
        (df["uids_asociadas_overlap_inicioC2"].notnull())
    ]

    print(f"✅ Mergers conservados por Filtro 1 overlap_inicio_C2: {len(df_conservados_overlap_inicioC2)}")
    display(df_conservados_overlap_inicioC2[[
        "uid_infall", "grupo", "uid_disruption", "uid_disruption_overlap_inicioC2",
        "uids_asociadas_overlap_inicioC2", "mass_stars_infall",
        "frac_star_A_en_Bs_overlap_inicioC2", "frac_star_Bs_en_A_overlap_inicioC2",
        "mass_gas_infall",
        "frac_gas_A_en_Bs_overlap_inicioC2",  "frac_gas_Bs_en_A_overlap_inicioC2",
        "mass_dm_infall",
        "frac_dm_A_en_Bs_overlap_inicioC2",   "frac_dm_Bs_en_A_overlap_inicioC2"
    ]])

    df_eliminados_overlap_inicioC2 = df[df["flag_overlap_eliminar_inicioC2"] == True]

    print(f"❌ Mergers eliminados por Filtro 1 overlap_inicio_C2: {len(df_eliminados_overlap_inicioC2)}")
    display(df_eliminados_overlap_inicioC2[[
        "uid_infall", "grupo", "uid_disruption", "uid_overlap_conservado_inicioC2"
    ]])


    # ============================================================
    # FILTRO 2: Grupos con overlap donde el primer merger NO es C2,
    #           pero el grupo contiene al menos un merger C2
    # ============================================================
    # Este filtro identifica grupos de mergers con partículas en común, en los cuales
    # el merger más antiguo no pertenece al grupo C2, pero al menos uno de los mergers
    # posteriores sí es tipo C2.
    #
    # Lógica del filtro:
    # - Se conserva el merger C2 más antiguo (uid_conservado).
    # - Se comparan las partículas de este merger con los mergers posteriores (uid).
    # - Si un merger posterior (uid) está estructuralmente contenido en el C2
    #   (≥75% de la componente dominante y ≥50% en las secundarias con >100 partículas),
    #   entonces se marca como redundante y se elimina.
    # - Se guardan las fracciones de solapamiento y los UID de los mergers asociados.
    #
    # Resultado: se eliminan mergers redundantes posteriores si sus componentes están
    # bien contenidas en un C2 anterior.


    # Inicializar columnas en el DataFrame
    df["flag_overlap_eliminar_contieneC2"] = False
    df["uid_disruption_overlap_contieneC2"] = None
    df["uids_asociadas_overlap_contieneC2"] = None
    df["uid_overlap_conservado_contieneC2"] = None

    for comp in ["star", "gas", "dm"]:
        df[f"frac_{comp}_A_en_Bs_overlap_contieneC2"] = None  # % de A contenido en Bs
        df[f"frac_{comp}_Bs_en_A_overlap_contieneC2"] = None  # % de Bs contenido en A

    # Aplicar filtro por grupo de overlap
    for grupo in componentes_overlap:
        grupo_ordenado = sorted(grupo, key=lambda uid: split_unique_id(int(uid))[0])
        grupo_int = [int(uid) for uid in grupo_ordenado]

        # Buscar mergers del grupo C2
        grupo_labels = df[df["uid_infall"].isin(grupo_int)][["uid_infall", "grupo"]]
        uids_c2 = grupo_labels[grupo_labels["grupo"] == "C2"]["uid_infall"].tolist()

        ###-###
        uid_primero = grupo_int[0]

        # Obtener grupo del primer merger
        fila_primero = df[df["uid_infall"] == uid_primero]
        if fila_primero.empty:
            continue

        grupo_primero = fila_primero["grupo"].values[0]

        # Filtro: solo procesar si el primero NO es C2 y hay al menos un C2, 
        # y no todos los mergers del grupo son C2
        if grupo_primero == "C2" or not uids_c2 or grupo_labels["grupo"].eq("C2").all():
            continue


        #uid_conservado = int(uids_c2[0])  # primer merger C2 en el grupo
        uid_conservado = int(sorted(uids_c2, key=lambda uid: split_unique_id(int(uid))[0])[0])


        # Obtener partículas del merger conservado (A)
        p_star_A = set(dict_particulas_infall[str(uid_conservado)]["stars"])
        p_gas_A  = set(dict_particulas_infall[str(uid_conservado)]["gas"])
        p_dm_A   = set(dict_particulas_infall[str(uid_conservado)]["dm"])

        n_star_A, n_gas_A, n_dm_A = len(p_star_A), len(p_gas_A), len(p_dm_A)

        uids_asociadas, uids_disruption, snaps = [], [], []
        fracs_star_A_en_Bs, fracs_gas_A_en_Bs, fracs_dm_A_en_Bs = [], [], []
        fracs_star_Bs_en_A, fracs_gas_Bs_en_A, fracs_dm_Bs_en_A = [], [], []

        for uid in grupo_int:
            if uid == uid_conservado:
                continue  # no compararse consigo mismo

            # Partículas del merger asociado (B)
            p_star_B = set(dict_particulas_infall[str(uid)]["stars"])
            p_gas_B  = set(dict_particulas_infall[str(uid)]["gas"])
            p_dm_B   = set(dict_particulas_infall[str(uid)]["dm"])

            n_star_B, n_gas_B, n_dm_B = len(p_star_B), len(p_gas_B), len(p_dm_B)

            # Obtener masas de B desde el DataFrame
            row_b = df[df["uid_infall"] == uid]
            if row_b.empty:
                continue

            mass_b = {
                "stars": row_b["mass_stars_infall"].values[0],
                "gas": row_b["mass_gas_infall"].values[0],
                "dm": row_b["mass_dm_infall"].values[0],
            }
            comp_dom_b = max(mass_b, key=mass_b.get)

            # === Fracciones ===
            # A en B
            frac_star_A_in_B = 100 * len(p_star_A & p_star_B) / n_star_A if n_star_A > 0 else None
            frac_gas_A_in_B  = 100 * len(p_gas_A  & p_gas_B)  / n_gas_A  if n_gas_A  > 0 else None
            frac_dm_A_in_B   = 100 * len(p_dm_A   & p_dm_B)   / n_dm_A   if n_dm_A  > 0 else None

            # B en A
            frac_star_B_in_A = 100 * len(p_star_A & p_star_B) / n_star_B if n_star_B > 0 else None
            frac_gas_B_in_A  = 100 * len(p_gas_A  & p_gas_B)  / n_gas_B  if n_gas_B  > 0 else None
            frac_dm_B_in_A   = 100 * len(p_dm_A   & p_dm_B)   / n_dm_B   if n_dm_B  > 0 else None

            fracs_star_A_en_Bs.append(frac_star_A_in_B)
            fracs_gas_A_en_Bs.append(frac_gas_A_in_B)
            fracs_dm_A_en_Bs.append(frac_dm_A_in_B)

            fracs_star_Bs_en_A.append(frac_star_B_in_A)
            fracs_gas_Bs_en_A.append(frac_gas_B_in_A)
            fracs_dm_Bs_en_A.append(frac_dm_B_in_A)

            # === Validación física: contenido estructural ===
            fracs_B_en_A = {
                "stars": frac_star_B_in_A,
                "gas": frac_gas_B_in_A,
                "dm": frac_dm_B_in_A,
            }

            # Dominante debe estar ≥75% en A
            if fracs_B_en_A[comp_dom_b] is None or fracs_B_en_A[comp_dom_b] < 75:
                continue

            # Secundarias con >100 partículas deben estar ≥50%
            cumple_secundarias = True
            for comp in ["stars", "gas", "dm"]:
                if comp == comp_dom_b:
                    continue
                n_part_B = len(dict_particulas_infall[str(uid)][comp])
                frac_in = fracs_B_en_A[comp]
                if n_part_B > 100 and (frac_in is None or frac_in < 50):
                    cumple_secundarias = False
                    break

            if not cumple_secundarias:
                continue

            # Marca B como eliminable
            df.loc[df["uid_infall"] == uid, "flag_overlap_eliminar_contieneC2"] = True
            df.loc[df["uid_infall"] == uid, "uid_overlap_conservado_contieneC2"] = uid_conservado
            uids_asociadas.append(uid)

            # Guardar UID de disrupción
            uid_dis = row_b["uid_disruption"].values[0]
            uids_disruption.append(uid_dis)
            snaps.append(split_unique_id(int(uid_dis))[0])

        # Guardar info en el merger conservado (A)
        if uids_asociadas:
            idx_max = snaps.index(max(snaps))
            df.loc[df["uid_infall"] == uid_conservado, "uid_disruption_overlap_contieneC2"] = uids_disruption[idx_max]
            df.loc[df["uid_infall"] == uid_conservado, "uids_asociadas_overlap_contieneC2"] = str(uids_asociadas)

            df.loc[df["uid_infall"] == uid_conservado, "frac_star_A_en_Bs_overlap_contieneC2"] = str(fracs_star_A_en_Bs)
            df.loc[df["uid_infall"] == uid_conservado, "frac_gas_A_en_Bs_overlap_contieneC2"]  = str(fracs_gas_A_en_Bs)
            df.loc[df["uid_infall"] == uid_conservado, "frac_dm_A_en_Bs_overlap_contieneC2"]   = str(fracs_dm_A_en_Bs)

            df.loc[df["uid_infall"] == uid_conservado, "frac_star_Bs_en_A_overlap_contieneC2"] = str(fracs_star_Bs_en_A)
            df.loc[df["uid_infall"] == uid_conservado, "frac_gas_Bs_en_A_overlap_contieneC2"]  = str(fracs_gas_Bs_en_A)
            df.loc[df["uid_infall"] == uid_conservado, "frac_dm_Bs_en_A_overlap_contieneC2"]   = str(fracs_dm_Bs_en_A)

    # ============================================================
    # RESUMEN DE RESULTADOS: Filtro 2 - overlap_contiene_C2
    # ============================================================
    # Este bloque resume el segundo filtro de limpieza por overlaps fuertes.
    # Aquí se identifican grupos donde el primer merger NO es tipo C2, pero
    # el grupo contiene al menos un merger C2. Se conserva el C2 más antiguo
    # y se eliminan mergers posteriores si están estructuralmente contenidos.
    # Se reportan:
    # - Los mergers C2 conservados y los mergers absorbidos.
    # - Las fracciones de partículas que justifican la absorción.
    # - Los mergers eliminados y el C2 que los reemplaza.


    df_conservados_overlap_contieneC2 = df[
        (df["flag_overlap_eliminar_contieneC2"] == False) &
        (df["uids_asociadas_overlap_contieneC2"].notnull())
    ]

    print(f"✅ Mergers conservados por Filtro 2 overlap_contiene_C2: {len(df_conservados_overlap_contieneC2)}")
    display(df_conservados_overlap_contieneC2[[
        "uid_infall", "grupo", "uid_disruption", "uid_disruption_overlap_contieneC2",
        "uids_asociadas_overlap_contieneC2", "mass_stars_infall",
        "frac_star_A_en_Bs_overlap_contieneC2", "frac_star_Bs_en_A_overlap_contieneC2",
        "mass_gas_infall",
        "frac_gas_A_en_Bs_overlap_contieneC2",  "frac_gas_Bs_en_A_overlap_contieneC2",
        "mass_dm_infall",
        "frac_dm_A_en_Bs_overlap_contieneC2",   "frac_dm_Bs_en_A_overlap_contieneC2"
    ]])

    df_eliminados_overlap_contieneC2 = df[df["flag_overlap_eliminar_contieneC2"] == True]

    print(f"❌ Mergers eliminados por Filtro 2 overlap_contiene_C2: {len(df_eliminados_overlap_contieneC2)}")
    display(df_eliminados_overlap_contieneC2[[
        "uid_infall", "grupo", "uid_disruption", "uid_overlap_conservado_contieneC2"
    ]])


    # ============================================================
    # FILTRO 3 - overlap_sinC: Grupos con overlap fuerte sin mergers tipo C
    # ============================================================
    # Aplica a componentes conectadas por partículas que solo contienen mergers A2 y B2.
    # Si hay un único B2 y al menos un A2, se evalúa si el B2 puede absorber a los A2
    # más recientes según fracciones de partículas (componente dominante ≥75% y secundarias ≥50% si >100 partículas).
    # Si se cumple, se conserva el B2 más antiguo y se eliminan los A2 asociados.


    df["flag_overlap_eliminar_sinC"] = False
    df["uid_disruption_overlap_sinC"] = None
    df["uids_asociadas_overlap_sinC"] = None
    df["uid_overlap_conservado_sinC"] = None

    for comp in ["star", "gas", "dm"]:
        df[f"frac_{comp}_A_en_Bs_overlap_sinC"] = None
        df[f"frac_{comp}_Bs_en_A_overlap_sinC"] = None

    for grupo in componentes_overlap:
        grupo_ordenado = sorted(grupo, key=lambda uid: split_unique_id(int(uid))[0])
        grupo_int = [int(uid) for uid in grupo_ordenado]

        df_grupo = df[df["uid_infall"].isin(grupo_int)]
        grupos_presentes = df_grupo["grupo"].tolist()

        # Saltar si hay algún C
        if any(g in ["C1", "C2"] for g in grupos_presentes):
            continue

        uids_b2 = df_grupo[df_grupo["grupo"] == "B2"]["uid_infall"].tolist()
        uids_a2 = df_grupo[df_grupo["grupo"] == "A2"]["uid_infall"].tolist()

        if not uids_b2 or not uids_a2:
            continue  # debe haber al menos un A2 y un B2

        # Elegimos el B2 más temprano como candidato a conservar
        uid_b2 = sorted(uids_b2, key=lambda uid: split_unique_id(int(uid))[0])[0]
        p_star_B = set(dict_particulas_infall[str(uid_b2)]["stars"])
        p_gas_B  = set(dict_particulas_infall[str(uid_b2)]["gas"])
        p_dm_B   = set(dict_particulas_infall[str(uid_b2)]["dm"])

        n_star_B, n_gas_B, n_dm_B = len(p_star_B), len(p_gas_B), len(p_dm_B)

        row_b = df[df["uid_infall"] == uid_b2]
        mass_b = {
            "stars": row_b["mass_stars_infall"].values[0],
            "gas": row_b["mass_gas_infall"].values[0],
            "dm": row_b["mass_dm_infall"].values[0],
        }
        comp_dom_b = max(mass_b, key=mass_b.get)

        uids_asociadas = []
        uids_disruption = []
        snaps = []
        fracs_star_A_en_Bs, fracs_gas_A_en_Bs, fracs_dm_A_en_Bs = [], [], []
        fracs_star_Bs_en_A, fracs_gas_Bs_en_A, fracs_dm_Bs_en_A = [], [], []

        for uid_a2 in uids_a2:
            fila_a2 = df[df["uid_infall"] == uid_a2]
            if fila_a2.empty:
                continue
            p_star_A = set(dict_particulas_infall[str(uid_a2)]["stars"])
            p_gas_A  = set(dict_particulas_infall[str(uid_a2)]["gas"])
            p_dm_A   = set(dict_particulas_infall[str(uid_a2)]["dm"])

            n_star_A, n_gas_A, n_dm_A = len(p_star_A), len(p_gas_A), len(p_dm_A)

            # === Fracciones ===
            frac_star_A_in_B = 100 * len(p_star_A & p_star_B) / n_star_A if n_star_A > 0 else None
            frac_gas_A_in_B  = 100 * len(p_gas_A  & p_gas_B)  / n_gas_A  if n_gas_A  > 0 else None
            frac_dm_A_in_B   = 100 * len(p_dm_A   & p_dm_B)   / n_dm_A   if n_dm_A  > 0 else None

            frac_star_B_in_A = 100 * len(p_star_A & p_star_B) / n_star_B if n_star_B > 0 else None
            frac_gas_B_in_A  = 100 * len(p_gas_A  & p_gas_B)  / n_gas_B  if n_gas_B  > 0 else None
            frac_dm_B_in_A   = 100 * len(p_dm_A   & p_dm_B)   / n_dm_B   if n_dm_B  > 0 else None

            fracs_star_A_en_Bs.append(frac_star_A_in_B)
            fracs_gas_A_en_Bs.append(frac_gas_A_in_B)
            fracs_dm_A_en_Bs.append(frac_dm_A_in_B)

            fracs_star_Bs_en_A.append(frac_star_B_in_A)
            fracs_gas_Bs_en_A.append(frac_gas_B_in_A)
            fracs_dm_Bs_en_A.append(frac_dm_B_in_A)

            fracs_B_en_A = {
                "stars": frac_star_B_in_A,
                "gas": frac_gas_B_in_A,
                "dm": frac_dm_B_in_A,
            }

            # Dominante ≥75%
            if fracs_B_en_A[comp_dom_b] is None or fracs_B_en_A[comp_dom_b] < 75:
                continue

            cumple_secundarias = True
            for comp in ["stars", "gas", "dm"]:
                if comp == comp_dom_b:
                    continue
                n_part = len(dict_particulas_infall[str(uid_b2)][comp])
                frac_in = fracs_B_en_A[comp]
                if n_part > 100 and (frac_in is None or frac_in < 50):
                    cumple_secundarias = False
                    break

            if not cumple_secundarias:
                continue

            # Marca A2 como eliminable
            df.loc[df["uid_infall"] == uid_a2, "flag_overlap_eliminar_sinC"] = True
            df.loc[df["uid_infall"] == uid_a2, "uid_overlap_conservado_sinC"] = uid_b2

            uids_asociadas.append(uid_a2)
            uid_dis = fila_a2["uid_disruption"].values[0]
            uids_disruption.append(uid_dis)
            snaps.append(split_unique_id(int(uid_dis))[0])

        if uids_asociadas:
            idx_max = snaps.index(max(snaps))
            df.loc[df["uid_infall"] == uid_b2, "uid_disruption_overlap_sinC"] = uids_disruption[idx_max]
            df.loc[df["uid_infall"] == uid_b2, "uids_asociadas_overlap_sinC"] = str(uids_asociadas)

            df.loc[df["uid_infall"] == uid_b2, "frac_star_A_en_Bs_overlap_sinC"] = str(fracs_star_A_en_Bs)
            df.loc[df["uid_infall"] == uid_b2, "frac_gas_A_en_Bs_overlap_sinC"]  = str(fracs_gas_A_en_Bs)
            df.loc[df["uid_infall"] == uid_b2, "frac_dm_A_en_Bs_overlap_sinC"]   = str(fracs_dm_A_en_Bs)

            df.loc[df["uid_infall"] == uid_b2, "frac_star_Bs_en_A_overlap_sinC"] = str(fracs_star_Bs_en_A)
            df.loc[df["uid_infall"] == uid_b2, "frac_gas_Bs_en_A_overlap_sinC"]  = str(fracs_gas_Bs_en_A)
            df.loc[df["uid_infall"] == uid_b2, "frac_dm_Bs_en_A_overlap_sinC"]   = str(fracs_dm_Bs_en_A)

    # ============================================================
    # RESUMEN DE RESULTADOS: Filtro 3 - overlap_sinC (A2 vs B2)
    # ============================================================
    # Este bloque resume el resultado del filtro aplicado a grupos con overlap fuerte
    # que no contienen mergers tipo C (solo A2 y B2). Se buscó conservar el merger B2 más
    # antiguo si sus partículas dominantes estaban suficientemente contenidas en los A2 más recientes.
    # Se reportan:
    # - Los mergers B2 conservados y los A2 que absorbieron.
    # - Las fracciones de partículas que justifican la absorción (por componente).
    # - Los mergers A2 eliminados junto con su B2 asociado.

    df_conservados_overlap_sinC = df[
        (df["flag_overlap_eliminar_sinC"] == False) & 
        (df["uids_asociadas_overlap_sinC"].notnull())
    ]
    df_eliminados_overlap_sinC = df[df["flag_overlap_eliminar_sinC"] == True]

    print(f"✅ Mergers conservados por Filtro 3 overlap_sinC (A2 vs B2): {len(df_conservados_overlap_sinC)}")
    display(df_conservados_overlap_sinC[[ 
        "uid_infall", "grupo", "uid_disruption", "uid_disruption_overlap_sinC",
        "uids_asociadas_overlap_sinC", "mass_stars_infall",
        "frac_star_A_en_Bs_overlap_sinC", "frac_star_Bs_en_A_overlap_sinC",
        "mass_gas_infall", "frac_gas_A_en_Bs_overlap_sinC", "frac_gas_Bs_en_A_overlap_sinC",
        "mass_dm_infall", "frac_dm_A_en_Bs_overlap_sinC", "frac_dm_Bs_en_A_overlap_sinC"
    ]])

    print(f"❌ Mergers eliminados por Filtro 3 overlap_sinC (A2 vs B2): {len(df_eliminados_overlap_sinC)}")
    display(df_eliminados_overlap_sinC[[ 
        "uid_infall", "grupo", "uid_disruption", "uid_overlap_conservado_sinC"
    ]])


    # ============================================================
    # FILTRO 4 (POST-OVERLAPS): Revisión de componente dominante en snapshots contiguos para mergers A2 y B2
    # ============================================================
    # Este paso se aplica únicamente a mergers clasificados como A2 o B2 que **no han sido eliminados** por los filtros
    # de overlaps previos (filtros 1 a 3). El objetivo es detectar si estas estructuras corresponden en realidad a
    # subdivisiones espurias dentro de la galaxia central (para A2) o fusiones instantáneas por resolución limitada (para B2).
    #
    # Para cada merger:
    #
    # - Si es A2: se analiza el snapshot anterior a su nacimiento (`snap_infall - 1`).
    # - Si es B2: se analiza el snapshot posterior a su disrupción (`snap_infall + 1`).
    #
    # En ese snapshot contiguo se identifica el subhalo **dominante** entre todas las partículas del merger.
    # Luego se calcula su `uid_dominante` y se verifica si ese objeto es parte de la main branch de la galaxia central.
    #
    # - Para A2:
    #     - Si `uid_dominante` está en la main branch de la central → ❌ el merger se elimina 
    #       (`flag_eliminar_A_snap_prev_dominante_es_central = True`).
    #       Esto indica que la estructura A2 no es realmente nueva, sino una subdivisión espuria del código.
    #
    # - Para B2:
    #     - Si `uid_dominante` está en la main branch de la central → ✅ el merger se conserva explícitamente
    #       (`flag_conservar_B_snap_post_dominante_es_central = True`), ya que probablemente se fusionó en un solo snapshot
    #       con la central y `snap_infall == snap_disruption` no es un error sino un límite de resolución temporal.
    #
    # - Si la mayoría de las partículas están unbound (`SubFindNumber == -1`) en el snapshot contiguo → 
    #     el merger no se elimina (se asume válida la detección).
    #
    # También se calculan las fracciones de partículas de cada tipo (estrellas, gas y DM) del merger original que terminan
    # en el subhalo dominante (`frac_star_en_dominante`, `frac_gas_en_dominante`, `frac_dm_en_dominante`), lo cual permite
    # evaluar cuán consistente es la conexión entre el merger y la estructura dominante identificada.




    rama_central = limpiar_main_branch(main_branch_completa(str(uid_target)))

    # 1. Obtener todos los snapshots válidos (los disponibles en el hdf5)
    with h5py.File(path_hdf5, "r") as f:
        snaps_validos = sorted([
            int(k.replace("SnapNumber_", ""))
            for k in f.keys() if k.startswith("SnapNumber_")
        ])

    print("Snapshots válidos encontrados:", snaps_validos)

    # 2. Encontrar el mínimo contiguo desde el máximo hacia abajo
    snap_max = snaps_validos[-1]
    snap_min_contiguo = snap_max

    for s in reversed(snaps_validos[:-1]):
        if s == snap_min_contiguo - 1:
            snap_min_contiguo = s
        else:
            break  # se rompe la continuidad hacia atrás

    def analizar_origen_o_destino(uid_infall, grupo, df, tree_1, tree_2, path_hdf5, uid_central, dict_particulas_infall, uid_disruption):
        from collections import Counter

        snap_infall, _ = split_unique_id(int(uid_infall))
        uid_infall_str = str(uid_infall)

        # Obtener partículas
        part_dict = dict_particulas_infall.get(uid_infall_str, {})
        ids_star = part_dict.get("stars", np.array([], dtype=np.uint64))
        ids_gas = part_dict.get("gas", np.array([], dtype=np.uint64))
        ids_dm = part_dict.get("dm", np.array([], dtype=np.uint64))
        part_ids_all = np.concatenate([ids_star, ids_gas, ids_dm])

        if len(part_ids_all) == 0:
            return pd.DataFrame([{
                "uid_infall": uid_infall, "uid_disruption": uid_disruption, "grupo": grupo, "snap": None, "uid_dominante": None, "snap_dominante":None,
                "flag_resultado": "SIN PARTICULAS", "frac_star_en_dominante": None,
                "frac_gas_en_dominante": None, "frac_dm_en_dominante": None
            }])

        resultados = []
        snap = max(snap_infall - 1, snap_min_contiguo) if grupo == "A2" else snap_infall + 1 # si grupo es A buscamos en el snap previo a la formación de la rama, si es B en el snap posterior a la muerte de la rama

        with h5py.File(path_hdf5, "r") as f:
            g = f[f"SnapNumber_{snap}"]
            ids_all, subfinds_all, tipos = [], [], []

            for ptype, tipo in zip(["PartType0", "PartType1", "PartType4"], ["gas", "dm", "stars"]):
                if f"{ptype}/ParticleIDs" not in g:
                    continue

                ids = g[f"{ptype}/ParticleIDs"][:]
                subf = g[f"{ptype}/SubFindNumber"][:]
                mask_base = np.isin(ids, part_ids_all)

                if grupo == "B2" and tipo == "stars":
                    mask_extra = np.isin(ids, ids_gas)  # estrellas formadas desde gas
                    mask = mask_base | mask_extra
                else:
                    mask = mask_base

                ids_all.extend(ids[mask])
                subfinds_all.extend(subf[mask])
                tipos.extend([tipo] * np.sum(mask))


        subf_dominante = Counter(subfinds_all).most_common(1)[0][0]

        if subf_dominante == -1:
            return pd.DataFrame([{
                "uid_infall": uid_infall, "uid_disruption": uid_disruption, "grupo": grupo,
                "uid_dominante": -1, "snap_dominante":snap, "flag_resultado": "DOM UNBOUND",
                "frac_star_en_dominante": None,
                "frac_gas_en_dominante": None,
                "frac_dm_en_dominante": None
            }])

        uid_dominante = int(f"{snap:03d}{subf_dominante:06d}")

        def fraccion_en_dominante(ids_origen, tipos_all, subfinds_all, ids_all, tipo_str):
            ids_tipo_en_snap = [pid for pid, tipo in zip(ids_all, tipos_all) if tipo == tipo_str]
            subfinds_tipo_en_snap = [sf for sf, tipo in zip(subfinds_all, tipos_all) if tipo == tipo_str]
            subf_dict = dict(zip(ids_tipo_en_snap, subfinds_tipo_en_snap))

            ids_origen_set = set(ids_origen)
            ids_presentes = ids_origen_set.intersection(subf_dict.keys())
            conteo = sum(1 for pid in ids_presentes if subf_dict[pid] == subf_dominante)

            return round(conteo / len(ids_origen), 3) if len(ids_origen) > 0 else np.nan

        frac_star = fraccion_en_dominante(ids_star, tipos, subfinds_all, ids_all, "stars")
        frac_gas = fraccion_en_dominante(ids_gas, tipos, subfinds_all, ids_all, "gas")
        frac_dm = fraccion_en_dominante(ids_dm, tipos, subfinds_all, ids_all, "dm")

        # Si este uid ya está en la main branch de la central
        if uid_dominante == rama_central[snap]:

            return pd.DataFrame([{
                "uid_infall": uid_infall, "uid_disruption": uid_disruption, "grupo": grupo,
                "uid_dominante": uid_dominante, "snap_dominante":snap, "flag_resultado": "DOM ES CENTRAL",
                "frac_star_en_dominante": frac_star,
                "frac_gas_en_dominante": frac_gas,
                "frac_dm_en_dominante": frac_dm
            }])

        # Si no está en la main branch de la central
        return pd.DataFrame([{
            "uid_infall": uid_infall, "uid_disruption": uid_disruption, "grupo": grupo,
            "uid_dominante": uid_dominante, "snap_dominante":snap, "flag_resultado": "DOM NO ES CENTRAL",
            "frac_star_en_dominante": frac_star, "frac_gas_en_dominante": frac_gas, "frac_dm_en_dominante": frac_dm
        }])



    def correr_revision_A2_B2(df, tree_1, tree_2, path_hdf5, uid_central, dict_particulas_infall):
        # Sólo considerar mergers A2 o B2 sin flags de eliminación
        flags_exclusivos = [
            "flag_overlap_eliminar_inicioC2",
            "flag_overlap_eliminar_contieneC2",
            "flag_overlap_eliminar_sinC"
        ]
        df_sin_flag = df[
            (df["grupo"].isin(["A2", "B2"])) &
            (~df[flags_exclusivos].any(axis=1))
        ]


        # GUARD: si no hay A2/B2 que revisar, devolver DF vacío con el esquema esperado
        if df_sin_flag.empty:
            return pd.DataFrame(columns=[
                "uid_infall","uid_disruption","grupo","snap",
                "uid_dominante","snap_dominante","flag_resultado",
                "frac_star_en_dominante","frac_gas_en_dominante","frac_dm_en_dominante",
                "flag_conservar_B_snap_post_dominante_es_central",
                "flag_eliminar_A_snap_prev_dominante_es_central"
            ])

        resultados = []

        for _, row in tqdm(df_sin_flag.iterrows(), total=len(df_sin_flag)):
            uid_infall = row["uid_infall"]
            uid_disruption = row["uid_disruption"]
            grupo = row["grupo"]
            df_res = analizar_origen_o_destino(uid_infall, grupo, df, tree_1, tree_2, path_hdf5, uid_central, dict_particulas_infall, uid_disruption)

            # Marcar como eliminable si uid_dominante ya es parte de la central
            uid_dom = df_res.loc[0, "uid_dominante"]
            if grupo == 'A2':
                df_res["flag_conservar_B_snap_post_dominante_es_central"] = False
                if pd.notna(uid_dom) and int(uid_dom) in rama_central.values():
                    df_res["flag_eliminar_A_snap_prev_dominante_es_central"] = True
                else:
                    df_res["flag_eliminar_A_snap_prev_dominante_es_central"] = False

            elif grupo == 'B2':
                df_res["flag_eliminar_A_snap_prev_dominante_es_central"] = False
                if pd.notna(uid_dom) and int(uid_dom) in rama_central.values():
                    df_res["flag_conservar_B_snap_post_dominante_es_central"] = True
                else:
                    df_res["flag_conservar_B_snap_post_dominante_es_central"] = False

            resultados.append(df_res)

        # GUARD: 0 filas (o 1) → concat no truena
        if not resultados:
            return pd.DataFrame(columns=[
                "uid_infall","uid_disruption","grupo","snap",
                "uid_dominante","snap_dominante","flag_resultado",
                "frac_star_en_dominante","frac_gas_en_dominante","frac_dm_en_dominante",
                "flag_conservar_B_snap_post_dominante_es_central",
                "flag_eliminar_A_snap_prev_dominante_es_central"
            ])
        # Unir todas las filas individuales
        df_final = pd.concat(resultados, ignore_index=True)
        return df_final


    # === Correr revisión
    df_revision = correr_revision_A2_B2(df, tree_1, tree_2, path_hdf5, int(uid_target), dict_particulas_infall)

    # ============================================================
    # Resumen de resultados — FILTRO 4 (dominante en snapshot contiguo)
    # ============================================================

    # A2: eliminados si el dominante previo ya era parte de la central
    a2 = df_revision[df_revision["grupo"] == "A2"]

    a2_eliminados = a2["flag_eliminar_A_snap_prev_dominante_es_central"].sum()
    a2_totales = len(a2)
    a2_pendientes = a2_totales - a2_eliminados

    print("🔍 Filtro 4 — Mergers A2")
    print(f"    ❌ Eliminados (dominante ya era parte de la central): {a2_eliminados}")
    print(f"    🟡 Aún no eliminados: {a2_pendientes}")

    # B2: conservados si el dominante post ya es parte de la central
    b2 = df_revision[df_revision["grupo"] == "B2"]

    b2_conservados = b2["flag_conservar_B_snap_post_dominante_es_central"].sum()
    b2_totales = len(b2)
    b2_pendientes = b2_totales - b2_conservados

    print("\n🔍 Filtro 4 — Mergers B2")
    print(f"    ✅ Conservados (dominante ya era la central): {b2_conservados}")
    print(f"    🟡 Aún no marcados como conservar: {b2_pendientes}")


    # ============================================================
    # FILTRO 5 (POST-A2): Detección de ramas A2 espurias o redundantes
    # ============================================================
    # Este filtro se aplica a mergers del grupo A2 que **no han sido eliminados**
    # por los filtros anteriores. Se identifican dos posibles casos:
    #
    # 1. CASO "ALWAYS UNBOUND":
    #    Si el `uid_dominante` en el snapshot anterior al infall es -1, se
    #    revisa hacia atrás si alguna vez estuvo en una subestructura. 
    #    - Si **nunca** lo estuvo, se marca como `flag_eliminar_A2_always_unbound = True`.
    #    - Si se encuentra que sí perteneció a alguna subestructura, se guarda
    #      el nuevo `uid_dominante_2` y se continúa con el análisis.
    #
    # 2. CASO "YA CAPTURADO POR OTRA RAMA":
    #    Si el `uid_dominante` (original o corregido) se fusiona con la galaxia
    #    central **posteriormente** como parte de otra rama (detectada en `df`),
    #    y esa otra rama fue clasificada como tipo C2, entonces:
    #    - Si el snapshot de infall del merger A2 es el mismo que el de la rama C2:
    #        → `flag_eliminar_A2_suma_a_C2 = True` (sus masas deben sumarse).
    #    - Si el snapshot de infall del merger A2 es posterior:
    #        → `flag_eliminar_A2_ya_capturado = True` (fue incluida antes).
    #


    from collections import Counter
    def aplicar_filtro5_revision_A2(df, df_revision, tree_1, tree_2, path_hdf5, dict_particulas_infall, uid_target):
        rama_central = limpiar_main_branch(main_branch_completa(str(uid_target)))
        resultados = []

        df_a2 = df_revision[
            (df_revision["grupo"] == "A2") &
            (~df_revision["flag_eliminar_A_snap_prev_dominante_es_central"])
        ]

        for _, row in tqdm(df_a2.iterrows(), total=len(df_a2)):
            uid_a2 = int(row["uid_infall"])
            uid_disr_a2 = int(row["uid_disruption"])
            snap_infall, _ = split_unique_id(uid_a2)
            uid_dominante = row["uid_dominante"]

            # Inicializar resultado
            res = {
                "uid_infall": uid_a2,
                "uid_disruption": uid_disr_a2,
                "flag_eliminar_A2_ya_capturado": False,
                "flag_eliminar_A2_suma_a_C2": False,
                "flag_eliminar_A2_always_unbound": False,
                "uid_infall_conservado": None,
                "uid_disruption_conservado": None,
                "uid_dominante_2": None,
                "flag_rama_cortada": False,
                "flag_rama_cortada_rama": None        }

            # ---------------------------------------------------------------------
            # CASO 1: Dominante == -1 → buscar si alguna vez estuvo en subestructura
            # ---------------------------------------------------------------------
            if uid_dominante == -1:
                part_dict = dict_particulas_infall.get(str(uid_a2), {})
                ids_all = np.concatenate([
                    part_dict.get("stars", np.array([], dtype=np.uint64)),
                    part_dict.get("gas", np.array([], dtype=np.uint64)),
                    part_dict.get("dm", np.array([], dtype=np.uint64))
                ])

                encontrado = False
                for snap in range(max(snap_infall - 2, snap_min_contiguo), -1, -1):
                    with h5py.File(path_hdf5, "r") as f:
                        subfs = []
                        for ptype in ["PartType0", "PartType1", "PartType4"]:
                            if f"SnapNumber_{snap}/{ptype}/ParticleIDs" not in f:
                                continue
                            g = f[f"SnapNumber_{snap}"]
                            ids = g[f"{ptype}/ParticleIDs"][:]
                            subf = g[f"{ptype}/SubFindNumber"][:]
                            mask = np.isin(ids, ids_all)
                            subfs.extend(subf[mask])

                    subfs = [s for s in subfs if s != -1]
                    if len(subfs) > 0:
                        subf_dominante = Counter(subfs).most_common(1)[0][0]
                        uid_dominante_2 = int(f"{snap:03d}{subf_dominante:06d}")
                        if uid_dominante_2 != -1:
                            res["uid_dominante_2"] = uid_dominante_2
                            uid_dominante = uid_dominante_2
                            encontrado = True
                            break

                if not encontrado:
                    res["flag_eliminar_A2_always_unbound"] = True
                    resultados.append(res)
                    continue
                    
            if pd.isna(uid_dominante):
                # no hay dominante fiable → no se puede evaluar CASO 2
                resultados.append(res)
                continue


            # ---------------------------------------------------------------------
            # CASO 2: uid_dominante existe → ver si se fusiona luego con la central
            # ---------------------------------------------------------------------

            # Si el nodo no existe en ninguno de los trees, salto este A2 (evito KeyError)
            _cand_str = str(uid_dominante)
            try:
                _cand_int = int(uid_dominante)
            except Exception:
                _cand_int = None

            if ((_cand_str not in tree_1) and (_cand_str not in tree_2) and
                (_cand_int is None or ((_cand_int not in tree_1) and (_cand_int not in tree_2)))):
                # No hay rama en los árboles para evaluar → no forzar etiquetas ni romper
                resultados.append(res)
                continue

            # Si existe, intento construir la rama; si aún así falta el nodo en el grafo, salto
            try:
                rama_dom = limpiar_main_branch(main_branch_completa(str(uid_dominante)))
            except KeyError:
                resultados.append(res)
                continue

            dict_diff = {
                snap: rama_dom[snap]
                for snap in rama_central.keys() & rama_dom.keys()
                if rama_central[snap] != rama_dom[snap]
            }

            if int(uid_target) not in rama_dom.values():
                # Dom nunca se fusiona con la galaxia central o rama está cortada (más probable)
                if snap_target not in rama_dom.keys():  # rama cortada
                    res["flag_rama_cortada"] = True
                    res["flag_rama_cortada_rama"] = rama_dom
                resultados.append(res)
                continue

            # >>> GUARD: si no hay snaps distintos entre rama_dom y rama_central, no hay pivote.
            #     Interpretación conservadora: ya estaba capturado por la main branch.
            if len(dict_diff) == 0:
                res["flag_eliminar_A2_ya_capturado"] = True
                resultados.append(res)
                continue

            max_snap_diff = max(dict_diff.keys())
            valor = dict_diff[max_snap_diff]

            if valor in df["uid_disruption"].values:
                fila_candidata = df[df["uid_disruption"] == valor].iloc[0]
                grupo_cand = fila_candidata["grupo"]
                snap_infall_cand, _ = split_unique_id(int(fila_candidata["uid_infall"]))
                if grupo_cand == "C2":
                    res["uid_infall_conservado"] = fila_candidata["uid_infall"]
                    res["uid_disruption_conservado"] = fila_candidata["uid_disruption"]
                    if snap_infall_cand == snap_infall:
                        res["flag_eliminar_A2_suma_a_C2"] = True
                    elif snap_infall_cand < snap_infall:
                        res["flag_eliminar_A2_ya_capturado"] = True

            resultados.append(res)

        return pd.DataFrame(resultados)

    df_filtro5 = aplicar_filtro5_revision_A2(
        df=df,
        df_revision=df_revision,
        tree_1=tree_1,
        tree_2=tree_2,
        path_hdf5=path_hdf5,
        dict_particulas_infall=dict_particulas_infall,
        uid_target=uid_target
    )

    # Si df_filtro5 está vacío, crear DataFrame vacío con las columnas necesarias
    if df_filtro5.empty:
        df_filtro5 = pd.DataFrame(columns=[
            "uid_infall",
            "uid_disruption",
            "flag_eliminar_A2_ya_capturado",
            "flag_eliminar_A2_suma_a_C2",
            "flag_eliminar_A2_always_unbound",
            "uid_infall_conservado",
            "uid_disruption_conservado",
            "uid_dominante_2",
            "flag_rama_cortada",
            "flag_rama_cortada_rama"
        ])



    if (df_filtro5["flag_rama_cortada"] == True).any():

        # ============================================================
        # FILTRO 6 – Rama rota: Validación física usando partículas en uid_dominante_2
        # ============================================================
        # Verifica si mergers A2 marcados como "rama cortada" en el Filtro 5 deberían conservarse,
        # comparando sus partículas con el subhalo dominante posterior (`uid_dominante_2`).
        # Usa los mismos criterios físicos que los filtros de overlaps:
        # - ≥90% de la componente más masiva en `uid_dominante_2`
        # - Secundarias con >100 partículas: ≥50% también en esa estructura
        # ============================================================

        resultados_f6 = []

        # Subconjunto de ramas cortadas con uid_dominante_2 definido
        df_ramas_rotas = df_filtro5[
            (df_filtro5["flag_rama_cortada"] == True) &
            (df_filtro5["uid_dominante_2"].notnull()) &
            (df_filtro5["uid_dominante_2"] != -1)
        ].merge(
            df[["uid_infall", "grupo", "snap_infall", "mass_stars_infall", "mass_gas_infall", "mass_dm_infall"]],
            on="uid_infall", how="left"
        )

        for _, row in tqdm(df_ramas_rotas.iterrows(), total=len(df_ramas_rotas), desc="🔍 Filtro 6 – Validando ramas cortadas"):
            uid_infall = row["uid_infall"]
            uid_dominante_2 = int(row["uid_dominante_2"])
            snap2, subf2 = split_unique_id(uid_dominante_2)

            # Componente dominante
            masas = {
                "stars": row["mass_stars_infall"],
                "gas": row["mass_gas_infall"],
                "dm": row["mass_dm_infall"]
            }
            comp_dominante = max(masas, key=masas.get)

            # Partículas del merger
            ids_star = set(dict_particulas_infall[str(uid_infall)]["stars"])
            ids_gas  = set(dict_particulas_infall[str(uid_infall)]["gas"])
            ids_dm   = set(dict_particulas_infall[str(uid_infall)]["dm"])
            n_all = {
                "stars": len(ids_star),
                "gas": len(ids_gas),
                "dm": len(ids_dm)
            }

            fracs_en_dominante = {}

            with h5py.File(path_hdf5, "r") as f:
                g = f[f"SnapNumber_{snap2}"]
                ids_en_dominante = {}
                for part_type, tipo in zip(["PartType4", "PartType0", "PartType1"], ["stars", "gas", "dm"]):
                    if f"{part_type}/ParticleIDs" not in g:
                        ids_en_dominante[tipo] = set()
                        continue
                    ids = g[f"{part_type}/ParticleIDs"][:]
                    subf = g[f"{part_type}/SubFindNumber"][:]
                    ids_en_dominante[tipo] = set(ids[subf == subf2])

            # Corrección para estrellas ausentes: buscar en gas
            ids_star_missing = ids_star - ids_en_dominante["stars"]
            ids_star_found_as_gas = ids_star_missing & ids_en_dominante["gas"]
            ids_star_corrected = (ids_star & ids_en_dominante["stars"]) | ids_star_found_as_gas
            frac_star = round(len(ids_star_corrected) / n_all["stars"], 3) if n_all["stars"] > 0 else None

            # Gas y DM
            frac_gas = round(len(ids_gas & ids_en_dominante["gas"]) / n_all["gas"], 3) if n_all["gas"] > 0 else None
            frac_dm  = round(len(ids_dm  & ids_en_dominante["dm"])  / n_all["dm"],  3) if n_all["dm"] > 0 else None

            fracs_en_dominante = {
                "stars": frac_star,
                "gas": frac_gas,
                "dm": frac_dm
            }

            # Evaluación física
            dominante_ok = (
                fracs_en_dominante[comp_dominante] is not None and
                fracs_en_dominante[comp_dominante] >= 0.9
            )

            cumple_secundarias = True
            for comp in ["stars", "gas", "dm"]:
                if comp == comp_dominante:
                    continue
                n_part = n_all[comp]
                frac = fracs_en_dominante[comp]
                if n_part > 100 and (frac is None or frac < 0.5):
                    cumple_secundarias = False
                    break

            flag_conservar = dominante_ok and cumple_secundarias

            resultados_f6.append({
                "uid_infall": uid_infall,
                "uid_dominante_2": uid_dominante_2,
                "componente_dominante": comp_dominante,
                "frac_star_en_dominante_2": frac_star,
                "frac_gas_en_dominante_2": frac_gas,
                "frac_dm_en_dominante_2": frac_dm,
                "flag_conservar_rama_rota": flag_conservar
            })

        # Crear DataFrame
        df_filtro6 = pd.DataFrame(resultados_f6)

        # Resumen
        n_total = len(df_filtro6)
        n_conservados = df_filtro6["flag_conservar_rama_rota"].sum()
        n_no_conservados = n_total - n_conservados

        print("🔍 Filtro 6 — Rama rota (validación estructural)")
        print("===================================================")
        print(f"🧪 Total ramas A2 cortadas evaluadas        : {n_total}")
        print(f"✅ Reaparecen como estructura coherente      : {n_conservados}")
        print(f"❌ No muestran continuidad física suficiente : {n_no_conservados}")

    else:
        print("✅ Filtro 6 no se ejecuta: no hay ramas rotas que validar.")


    # ============================================================
    # FILTRO 7 - Bajo traspaso (<10% a galaxia central)
    # ============================================================

    # Crear copia base
    df_filtros_aplicados = df.copy()

    # Filtro 7 — Marcar como eliminar todos los mergers tipo X1 (sin traspaso significativo)
    df_filtros_aplicados["flag_eliminar_bajo_traspaso"] = (
        df_filtros_aplicados["grupo"].notna() &
        df_filtros_aplicados["grupo"].astype(str).str.endswith("1")
    )


    # ------------------------------------------------------------
    # FLAGS FINALES: CONSERVACIÓN Y ELIMINACIÓN TOTAL 
    # ------------------------------------------------------------

    # Este bloque final resume todos los filtros aplicados (1 a 7) en un único
    # DataFrame `df_filtros_aplicados`, determinando si cada merger debe:
    #   🔴 ser eliminado: `flag_eliminar_total = True`
    #   🟢 ser conservado: `flag_conservar_total = True`
    # También verifica posibles inconsistencias (ambos flags True).

    # ============================================================
    # PASO 0 – Definir df_filtros_aplicados como copia del original
    # ============================================================
    df_filtros_aplicados = df.copy()

    # ============================================================
    # PASO 1 – Filtro 7: Eliminar mergers tipo "1" (sin transferencia significativa)
    # ============================================================
    df_filtros_aplicados["flag_eliminar_bajo_traspaso"] = df_filtros_aplicados["grupo"].astype(str).str.endswith("1")
    df_filtros_aplicados["flag_no_transferencia"] = df_filtros_aplicados["flag_eliminar_bajo_traspaso"].copy()

    # ============================================================
    # PASO 2 – Filtros 1 a 3: Flags de overlaps
    # ============================================================
    flags_overlap = [
        "flag_overlap_eliminar_inicioC2",
        "flag_overlap_eliminar_contieneC2",
        "flag_overlap_eliminar_sinC"
    ]

    # Asegurar que estén presentes y sin NaNs
    for flag in flags_overlap:
        if flag not in df_filtros_aplicados.columns:
            df_filtros_aplicados[flag] = False
        else:
            df_filtros_aplicados[flag] = df_filtros_aplicados[flag].astype("boolean").fillna(False)

    # ============================================================
    # PASO 3 – Filtro 4: Flags desde df_revision (snapshot contiguo)
    # ============================================================
    flags_revision = [
        "flag_eliminar_A_snap_prev_dominante_es_central",
        "flag_conservar_B_snap_post_dominante_es_central"
    ]

    # Asegurar que estén presentes en df_revision
    for flag in flags_revision:
        if flag not in df_revision.columns:
            df_revision[flag] = False

    # Merge con df_filtros_aplicados
    df_filtros_aplicados = df_filtros_aplicados.merge(
        df_revision[["uid_infall"] + flags_revision],
        on="uid_infall", how="left"
    )

    # Reemplazar NaNs por False
    for flag in flags_revision:
        df_filtros_aplicados[flag] = df_filtros_aplicados[flag].astype("boolean").fillna(False)

    # ============================================================
    # PASO 4 – Filtro 5: Flags de rama espuria o ya capturada
    # ============================================================
    flags_eliminar_f5 = [
        "flag_eliminar_A2_ya_capturado",
        "flag_eliminar_A2_suma_a_C2",
        "flag_eliminar_A2_always_unbound"
    ]

    # Asegurar que estén en df_filtro5
    for flag in flags_eliminar_f5 + ["uid_infall_conservado"]:
        if flag not in df_filtro5.columns:
            df_filtro5[flag] = False if flag.startswith("flag") else None

    # Merge a df_filtros_aplicados
    df_filtros_aplicados = df_filtros_aplicados.merge(
        df_filtro5[["uid_infall"] + flags_eliminar_f5 + ["uid_infall_conservado"]],
        on="uid_infall", how="left"
    )

    # Rellenar NaNs
    for flag in flags_eliminar_f5:
        df_filtros_aplicados[flag] = df_filtros_aplicados[flag].astype("boolean").fillna(False)

    # ============================================================
    # PASO 5 – Filtro 6: Validación física de ramas cortadas
    # ============================================================
    if (df_filtro5["flag_rama_cortada"] == True).any():
        if "flag_conservar_rama_rota" in df_filtro6.columns:
            df_filtros_aplicados = df_filtros_aplicados.merge(
                df_filtro6[["uid_infall", "flag_conservar_rama_rota"]],
                on="uid_infall", how="left"
            )
            df_filtros_aplicados["flag_conservar_rama_rota"] = df_filtros_aplicados["flag_conservar_rama_rota"].astype("boolean").fillna(False)
        else:
            df_filtros_aplicados["flag_conservar_rama_rota"] = False
    else:
        df_filtros_aplicados["flag_conservar_rama_rota"] = False
    # ============================================================
    # PASO 6 – FLAG FINAL DE ELIMINACIÓN
    # ============================================================

    flags_eliminar_todos = (
        flags_overlap +
        ["flag_eliminar_bajo_traspaso"] +
        ["flag_eliminar_A_snap_prev_dominante_es_central"] +
        flags_eliminar_f5
    )

    # Si **cualquiera** de los flags anteriores es True → eliminar
    df_filtros_aplicados["flag_eliminar_total"] = (
        df_filtros_aplicados[flags_eliminar_todos].any(axis=1)
    )

    # ============================================================
    # PASO 7 – FLAG FINAL DE CONSERVACIÓN
    # ============================================================

    # Inicializar si no existe
    if "flag_conservar_total" not in df_filtros_aplicados.columns:
        df_filtros_aplicados["flag_conservar_total"] = False

    # Asegurar columna booleana
    df_filtros_aplicados["flag_conservar_total"] = (
        df_filtros_aplicados.get("flag_conservar_total", False)
        | df_filtros_aplicados["flag_conservar_B_snap_post_dominante_es_central"]
        | df_filtros_aplicados["flag_conservar_rama_rota"]
    )

    # ✅ Conservar mergers del grupo C2 que NO hayan sido marcados para eliminación
    es_grupo_C2 = df_filtros_aplicados["grupo"].astype(str).str.startswith("C2")
    df_filtros_aplicados["flag_conservar_total"] |= (
        es_grupo_C2 & (~df_filtros_aplicados["flag_eliminar_total"])
    )


    # ============================================================
    # PASO 8 – VERIFICACIÓN DE CONFLICTOS
    # ============================================================

    conflictos = df_filtros_aplicados[
        df_filtros_aplicados["flag_conservar_total"] &
        df_filtros_aplicados["flag_eliminar_total"]
    ]

    if len(conflictos) > 0:
        print("⚠️  Conflictos detectados: mergers con ambos flags True (conservar y eliminar):")
        display(conflictos[["uid_infall", "grupo"] + flags_eliminar_todos + ["flag_conservar_total"]])
    else:
        print("✅ Sin conflictos en flags finales.")

    # ============================================================
    # PASO 9 – MOTIVOS DE ELIMINACIÓN Y CONSERVACIÓN
    # ============================================================

    def extraer_flags_activos(row, lista_flags):
        return [flag for flag in lista_flags if row.get(flag, False)]

    # Motivos de eliminación: todos los flags que resultaron en True
    df_filtros_aplicados["motivo_eliminacion"] = df_filtros_aplicados.apply(
        lambda row: extraer_flags_activos(row, flags_eliminar_todos),
        axis=1
    )

    # Motivos de conservación: todos los flags de conservación que resultaron en True
    flags_conservacion = [
        "flag_conservar_B_snap_post_dominante_es_central",
        "flag_conservar_rama_rota"
    ]
    df_filtros_aplicados["motivo_conservacion"] = df_filtros_aplicados.apply(
        lambda row: extraer_flags_activos(row, flags_conservacion),
        axis=1
    )

    df_filtros_aplicados["motivo_eliminacion"] = df_filtros_aplicados["motivo_eliminacion"].apply(lambda x: ", ".join(x) if x else "")
    df_filtros_aplicados["motivo_conservacion"] = df_filtros_aplicados["motivo_conservacion"].apply(lambda x: ", ".join(x) if x else "")

    # ============================================================
    # FILTRO 8 – Fusiones post-infall no registradas en el merger tree
    # ============================================================
    # Este filtro identifica casos en que, después del `t_infall` del merger,
    # su rama principal se fusiona con otras estructuras (progenitores secundarios)
    # que no fueron registradas como mergers independientes.
    #
    # Estas fusiones post-infall pueden llevar masa adicional (estrellas, gas, DM)
    # que no está contabilizada en la masa original del merger (a `t_infall`).
    # Como resultado, la masa del merger estar subestimada.
    #
    # Lógica:
    # - Para cada merger del DataFrame:
    #     - Se recorre su rama principal desde `snap_disruption` hasta `snap_infall`.
    #     - En cada snapshot, se revisan los progenitores del nodo actual.
    #     - Si hay más de un progenitor en `snap - 1`, y alguno no pertenece a
    #       la main branch, se considera que hubo una fusión no registrada.
    #     - Se rastrea el UID correspondiente al `t_infall` de esa subestructura secundaria.
    #     - Se calcula la masa de estas estructuras en `t_disruption` y `t_infall`.
    #
    # Columnas agregadas:
    # - `uids_extra_disr`       → lista de UIDs de subhalos fusionados post-infall (en disrupción)
    # - `uids_extra_infall`     → lista de sus correspondientes UIDs a `t_infall`
    # - `mass_*_extra_disr`     → masa de las estructuras adicionales al momento de fusión
    # - `mass_*_extra_infall`   → masa de esas estructuras al momento de `t_infall`
    # - `mass_*_total_infall`   → masa total del merger + estructuras ocultas, usada para
    #                              evaluar eficiencia de transferencia corregida
    #
    # ============================================================

    def get_uid_at_snap(uid, snap_target):
        """
        Dado un UID y un snapshot target, retorna el UID correspondiente a ese snapshot
        en su rama principal completa (extendida), o None si no existe.
        """
        rama = limpiar_main_branch(main_branch_completa(uid))
        return str(rama.get(snap_target))  # str para mantener consistencia

    def obtener_masas_uid(uid, path_hdf5):
        snap, subf = split_unique_id(int(uid))
        with h5py.File(path_hdf5, "r") as f:
            snap_grp = f[f"SnapNumber_{snap}"]
            h = snap_grp["Header/HubbleParam"][()]
            mtable = snap_grp["Header/MassTable"][()]

            # === Offsets ===
            offset_s = snap_grp["SubGroups/PartType4/Offsets"][subf].astype(int)
            offset_g = snap_grp["SubGroups/PartType0/Offsets"][subf].astype(int)
            offset_dm = snap_grp["SubGroups/PartType1/Offsets"][subf].astype(int)

            # === Masas estelares ===
            if offset_s[1] > offset_s[0]:
                m_star = snap_grp["PartType4/Masses"][offset_s[0]:offset_s[1]].sum() * 1e10 / h
            else:
                m_star = 0.0

            # === Masas de gas ===
            if offset_g[1] > offset_g[0]:
                m_gas = snap_grp["PartType0/Masses"][offset_g[0]:offset_g[1]].sum() * 1e10 / h
            else:
                m_gas = 0.0

            # === Masas DM ===
            if offset_dm[1] > offset_dm[0]:
                n_dm = offset_dm[1] - offset_dm[0]
                m_dm = n_dm * mtable[1] * 1e10 / h
            else:
                m_dm = 0.0

        return m_star, m_gas, m_dm


    # === Cargar árbol ===
    G = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)
    G_inv = G.reverse(copy=True)

    # === Columnas a agregar ===
    df_filtros_aplicados["uids_extra_disr"] = None
    df_filtros_aplicados["uids_extra_infall"] = None
    df_filtros_aplicados["mass_star_extra_disr"] = 0.0
    df_filtros_aplicados["mass_gas_extra_disr"] = 0.0
    df_filtros_aplicados["mass_dm_extra_disr"] = 0.0
    df_filtros_aplicados["mass_star_extra_infall"] = 0.0
    df_filtros_aplicados["mass_gas_extra_infall"] = 0.0
    df_filtros_aplicados["mass_dm_extra_infall"] = 0.0
    df_filtros_aplicados["mass_star_total_infall"] = 0.0
    df_filtros_aplicados["mass_gas_total_infall"] = 0.0
    df_filtros_aplicados["mass_dm_total_infall"] = 0.0

    # >>> ADDED: columnas para caracterizar fusiones post-infall por merger
    df_filtros_aplicados["n_post_infall"] = 0
    df_filtros_aplicados["uids_post_infall"] = [[] for _ in range(len(df_filtros_aplicados))]
    df_filtros_aplicados["stellar_mass_ratio_post_infall"] = [[] for _ in range(len(df_filtros_aplicados))]
    df_filtros_aplicados["baryonic_mass_ratio_post_infall"] = [[] for _ in range(len(df_filtros_aplicados))]

    # === Recorrer el DataFrame ===
    for idx, row in tqdm(df_filtros_aplicados.iterrows(), total=len(df_filtros_aplicados), desc="Detectando fusiones"):
        uid_dis = str(row["uid_disruption"])
        uid_inf = str(row["uid_infall"])

        snap_dis, _ = split_unique_id(int(uid_dis))
        snap_inf, _ = split_unique_id(int(uid_inf))

        rama_uid = limpiar_main_branch(main_branch_completa(uid_dis))
        uids_disr_extra, uids_infall_extra = [], []
        mass_star_disr, mass_gas_disr, mass_dm_disr = 0.0, 0.0, 0.0
        mass_star_inf,  mass_gas_inf,  mass_dm_inf  = 0.0, 0.0, 0.0

        for snap in range(snap_dis, snap_inf, -1):
            uid_actual = str(rama_uid[snap])
            if uid_actual not in G_inv:
                continue

            progenitores = list(G_inv.predecessors(uid_actual))
            progenitores_snap_anterior = [
                u for u in progenitores if split_unique_id(int(u))[0] == snap - 1
            ]

            if len(progenitores_snap_anterior) > 1:
                for u in progenitores_snap_anterior:
                    u_int = int(u)
                    es_main = u_int in rama_uid.values()
                    if es_main:
                        continue

                    # UID a infall
                    uid_infall_extra = get_uid_at_snap(u, snap_inf)
                    if uid_infall_extra is None:
                        continue

                    # Masas en disrupción
                    try:
                        m_star_d, m_gas_d, m_dm_d = obtener_masas_uid(u, path_hdf5)
                        mass_star_disr += m_star_d
                        mass_gas_disr  += m_gas_d
                        mass_dm_disr   += m_dm_d
                        uids_disr_extra.append(u)
                    except:
                        continue

                    # Masas en infall
                    try:
                        m_star_i, m_gas_i, m_dm_i = obtener_masas_uid(uid_infall_extra, path_hdf5)
                        mass_star_inf += m_star_i
                        mass_gas_inf  += m_gas_i
                        mass_dm_inf   += m_dm_i
                        uids_infall_extra.append(uid_infall_extra)
                    except:
                        continue
        # >>> ADDED: ratios de masa por cada fusión post-infall detectada
        # Denominador: masa de la galaxia principal al snap_infall de ESTE merger
        try:
            m_main_star_inf, m_main_gas_inf, _ = obtener_masas_uid(uid_inf, path_hdf5)
        except Exception as e:
            m_main_star_inf, m_main_gas_inf = 0.0, 0.0
            print(f"[!] No se pudo leer masa de la central en infall (uid_inf={uid_inf}): {e}")

        ratios_star, ratios_bary = [], []
        for uid_sat_inf in uids_infall_extra:
            try:
                ms, mg, _ = obtener_masas_uid(uid_sat_inf, path_hdf5)
            except Exception as e:
                print(f"[!] No se pudo leer masa del satélite {uid_sat_inf} en infall: {e}")
                ms, mg = 0.0, 0.0

            # Razón estelar y bariónica a t_infall
            r_star = (ms / m_main_star_inf) if m_main_star_inf > 0 else np.nan
            denom_b = (m_main_star_inf + m_main_gas_inf)
            r_bary = ((ms + mg) / denom_b) if denom_b > 0 else np.nan

            ratios_star.append(r_star)
            ratios_bary.append(r_bary)

        # Persistir en el dataframe por MERGER
        df_filtros_aplicados.at[idx, "n_post_infall"] = len(uids_infall_extra)
        df_filtros_aplicados.at[idx, "uids_post_infall"] = uids_infall_extra
        df_filtros_aplicados.at[idx, "stellar_mass_ratio_post_infall"] = ratios_star
        df_filtros_aplicados.at[idx, "baryonic_mass_ratio_post_infall"] = ratios_bary

        # Defaults cuando no hubo fusiones post-infall
        if len(uids_infall_extra) == 0:
            df_filtros_aplicados.at[idx, "n_post_infall"] = 0
            df_filtros_aplicados.at[idx, "uids_post_infall"] = []
            df_filtros_aplicados.at[idx, "stellar_mass_ratio_post_infall"] = []
            df_filtros_aplicados.at[idx, "baryonic_mass_ratio_post_infall"] = []


        # Guardar en el DataFrame
        df_filtros_aplicados.at[idx, "uids_extra_disr"] = uids_disr_extra
        df_filtros_aplicados.at[idx, "uids_extra_infall"] = uids_infall_extra
        df_filtros_aplicados.at[idx, "mass_star_extra_disr"] = mass_star_disr
        df_filtros_aplicados.at[idx, "mass_gas_extra_disr"] = mass_gas_disr
        df_filtros_aplicados.at[idx, "mass_dm_extra_disr"] = mass_dm_disr
        df_filtros_aplicados.at[idx, "mass_star_extra_infall"] = mass_star_inf
        df_filtros_aplicados.at[idx, "mass_gas_extra_infall"] = mass_gas_inf
        df_filtros_aplicados.at[idx, "mass_dm_extra_infall"] = mass_dm_inf

        # Sumar con la masa de la main branch (en snap_inf)
        try:
            m_main_star, m_main_gas, m_main_dm = obtener_masas_uid(uid_inf, path_hdf5)
            df_filtros_aplicados.at[idx, "mass_star_total_infall"] = m_main_star + mass_star_inf
            df_filtros_aplicados.at[idx, "mass_gas_total_infall"]  = m_main_gas  + mass_gas_inf
            df_filtros_aplicados.at[idx, "mass_dm_total_infall"]   = m_main_dm   + mass_dm_inf
        except Exception as e:
            df_filtros_aplicados.at[idx, "mass_star_total_infall"] = mass_star_inf
            df_filtros_aplicados.at[idx, "mass_gas_total_infall"]  = mass_gas_inf
            df_filtros_aplicados.at[idx, "mass_dm_total_infall"]   = mass_dm_inf
            print(f"❗Error al obtener masas para uid_inf={uid_inf}: {e}")


    # Filtrar filas donde `uids_extra_infall` no es una lista vacía
    cond_extra_infall = df_filtros_aplicados["uids_extra_infall"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    df_extra_infall_no_vacio = df_filtros_aplicados[cond_extra_infall].copy()

    print(f"🔍 Se encontraron {len(df_extra_infall_no_vacio)} mergers con fusiones post-infall no vacías.")

    # Filtrar los mergers eliminados por suma a C2 (flag del filtro 5)
    df_suma_c2 = df_filtros_aplicados[df_filtros_aplicados["flag_eliminar_A2_suma_a_C2"]].copy()

    # Paso 2: Agrupar los uid_infall de los mergers A2 eliminados, asociados a cada merger conservado (uid_infall_conservado)
    from collections import defaultdict

    dict_suma_filtro5 = defaultdict(list)

    for _, row in df_suma_c2.iterrows():
        uid_conservado = int(row["uid_infall_conservado"])
        uid_fuente = row["uid_infall"]

        if pd.notnull(uid_conservado):
            dict_suma_filtro5[str(uid_conservado)].append(str(uid_fuente))

    # Paso 3: Crear la nueva columna en el DataFrame principal
    df_filtros_aplicados["uids_extra_infall_filtro5"] = df_filtros_aplicados["uid_infall"].astype(str).map(dict_suma_filtro5)

    # Asegurarse de que las celdas sin entradas queden como listas vacías (para consistencia)
    df_filtros_aplicados["uids_extra_infall_filtro5"] = df_filtros_aplicados["uids_extra_infall_filtro5"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    print("✅ Columna 'uids_extra_infall_filtro5' generada correctamente.")

    # Mostrar filas donde la lista de uids extra no esté vacía (filtro 5)
    df_no_vacio = df_filtros_aplicados[df_filtros_aplicados["uids_extra_infall_filtro5"].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )]
    print(df_no_vacio[["uid_infall", "uids_extra_infall_filtro5"]])


    # ========================================
    # ACTUALIZACIÓN DE MASAS TOTALES EN INFALL
    # Incluir aporte de uids_extra_infall_filtro5
    # con print cuando se sume masa
    # ========================================

    for idx, row in tqdm(df_filtros_aplicados.iterrows(), total=len(df_filtros_aplicados), desc="Sumando masa de filtro 5"):
        uids_extra = row["uids_extra_infall_filtro5"]

        extra_star = 0.0
        extra_gas  = 0.0
        extra_dm   = 0.0

        for uid in uids_extra:
            try:
                m_star, m_gas, m_dm = obtener_masas_uid(uid, path_hdf5)
                extra_star += m_star
                extra_gas  += m_gas
                extra_dm   += m_dm
            except Exception as e:
                print(f"⚠️ Error al obtener masa para UID {uid}: {e}")
                continue

        # Si hay alguna masa adicional, hacer print
        if extra_star > 0 or extra_gas > 0 or extra_dm > 0:
            print(f"➕ idx {idx} — UID central {row['uid_infall']}")
            print(f"    → uids extra: {uids_extra}")
            print(f"    → M_extra = estrellas: {extra_star:.2e}, gas: {extra_gas:.2e}, DM: {extra_dm:.2e}")

        # Sumar a las columnas existentes (que ya tienen aporte del filtro 8)
        df_filtros_aplicados.at[idx, "mass_star_total_infall"] += extra_star
        df_filtros_aplicados.at[idx, "mass_gas_total_infall"]  += extra_gas
        df_filtros_aplicados.at[idx, "mass_dm_total_infall"]   += extra_dm


    print(f"\n📊 === RESUMEN FINAL DE DEPURACIÓN PARA {sim_name}-{target_subfind_id} ===\n")

    # Total de mergers
    total_mergers = len(df_filtros_aplicados)
    print(f"🔹 Total de mergers analizados: {total_mergers}")

    # ----------------------
    # Eliminados
    # ----------------------
    eliminados = df_filtros_aplicados[df_filtros_aplicados["flag_eliminar_total"]]
    n_eliminados = len(eliminados)
    print(f"\n❌ Mergers eliminados: {n_eliminados}")

    # Desglose por flag de eliminación
    flags_eliminar = [col for col in df_filtros_aplicados.columns if col.startswith("flag_eliminar_")]
    for flag in flags_eliminar:
        count = df_filtros_aplicados[flag].sum()
        if count > 0:
            print(f"   - {flag}: {count}")

    # ----------------------
    # Conservados
    # ----------------------
    conservados = df_filtros_aplicados[df_filtros_aplicados["flag_conservar_total"] & ~df_filtros_aplicados["flag_eliminar_total"]]
    n_conservados = len(conservados)
    print(f"\n✅ Mergers conservados: {n_conservados}")

    # Motivos de conservación
    flags_conservar = [col for col in df_filtros_aplicados.columns if col.startswith("flag_conservar_") and col != "flag_conservar_total"]
    for flag in flags_conservar:
        count = df_filtros_aplicados[flag].sum()
        if count > 0:
            print(f"   - {flag}: {count}")

    # ----------------------
    # Masa modificada por fusiones ocultas (filtro 8)
    # ----------------------
    modificados_f8 = df_filtros_aplicados["uids_extra_infall"].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
    print(f"\n🔄 Mergers con masa modificada por fusiones post-infall (filtro 8): {modificados_f8}")

    # Masa modificada por suma de mergers A2 a C2 (filtro 5)
    modificados_f5 = df_filtros_aplicados["uids_extra_infall_filtro5"].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
    print(f"🔄 Mergers con masa modificada por suma de mergers eliminados A2 (filtro 5): {modificados_f5}")

    print("\n✅ Resumen completo generado.")


    # ========================
    # CREAR TABLA FINAL BONITA
    # ========================

    # Filtrar mergers conservados
    df_conservados = df_filtros_aplicados[df_filtros_aplicados["flag_conservar_total"]].copy()

    # Determinar si la masa fue modificada por filtro 5 o filtro 8
    df_conservados["flag_masa_modificada"] = (
        df_conservados["uids_extra_infall"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False) |
        df_conservados["uids_extra_infall_filtro5"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    )

    # Determinar snap_infall y snap_disruption desde los uids
    df_conservados["snap_infall"] = df_conservados["uid_infall"].apply(lambda x: int(x) // int(1e6))
    df_conservados["snap_disruption"] = df_conservados["uid_disruption"].apply(lambda x: int(x) // int(1e6))

    # Flags adicionales de motivo de conservación (cuando grupo != C2)
    df_conservados["grupo"] = df_conservados["grupo"].astype(str)
    df_conservados["flag_conservado_noC2"] = (df_conservados["grupo"] != "C2")

    # ------------------------------------------------------------
    # Agregar columnas uid_disruption_new, motivo y uids asociadas
    # ------------------------------------------------------------

    def detectar_disruption_distinta(row):
        uid_d = row["uid_disruption"]
        for col_uid, motivo in [
            ("uid_disruption_overlap_inicioC2", "inicioC2"),
            ("uid_disruption_overlap_contieneC2", "contieneC2"),
            ("uid_disruption_overlap_sinC", "sinC")
        ]:
            if col_uid in row and pd.notnull(row[col_uid]) and row[col_uid] != uid_d:
                col_uids_asociadas = col_uid.replace("uid_disruption_overlap", "uids_asociadas_overlap")
                uids_asociadas = row.get(col_uids_asociadas, [])
                return pd.Series([row[col_uid], motivo, uids_asociadas])
        return pd.Series([None, None, None])

    #df_conservados[["uid_disruption_new", "motivo_disruption_cambia", "uids_asociadas_overlap"]] = \
    #    df_conservados.apply(detectar_disruption_distinta, axis=1)
    # --- reemplaza la asignación frágil por una expansión robusta ---
    # PATCH: expansión robusta de la salida de detectar_disruption_distinta
    def _as_tripleta(x):
        # Normaliza a [uid_disruption_new, motivo_disruption_cambia, uids_asociadas_overlap]
        if isinstance(x, (list, tuple)) and len(x) == 3:
            return list(x)
        if isinstance(x, dict):
            return [x.get("uid_disruption_new", pd.NA),
                    x.get("motivo_disruption_cambia", pd.NA),
                    x.get("uids_asociadas_overlap", [])]
        if isinstance(x, pd.Series) and len(x) == 3:
            return x.to_list()
        # Todo lo demás (None, escalar, longitudes incorrectas)
        return [pd.NA, pd.NA, []]

    # Nada que asignar: crea columnas vacías con tipos útiles y sigue
    if df_conservados.empty:
        df_conservados["uid_disruption_new"]       = pd.Series(dtype="Int64")
        df_conservados["motivo_disruption_cambia"] = pd.Series(dtype="string")
        df_conservados["uids_asociadas_overlap"]   = pd.Series(dtype="object")
    else:
        res_raw = df_conservados.apply(detectar_disruption_distinta, axis=1)
        tripletas = [_as_tripleta(x) for x in res_raw]
        res_df = pd.DataFrame(tripletas, index=df_conservados.index,
                              columns=["uid_disruption_new","motivo_disruption_cambia","uids_asociadas_overlap"])
        # Tipos/coherencia
        res_df["uid_disruption_new"]       = pd.to_numeric(res_df["uid_disruption_new"], errors="coerce").astype("Int64")
        res_df["motivo_disruption_cambia"] = res_df["motivo_disruption_cambia"].astype("string")
        res_df["uids_asociadas_overlap"]   = res_df["uids_asociadas_overlap"].apply(
            lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
        )
        df_conservados[["uid_disruption_new","motivo_disruption_cambia","uids_asociadas_overlap"]] = res_df


    # Convertir uid_disruption_new a entero nullable (Int64)
    df_conservados["uid_disruption_new"] = pd.to_numeric(df_conservados["uid_disruption_new"], errors='coerce').astype("Int64")


    # ------------------------------------------------------------
    # Selección y orden final de columnas
    # ------------------------------------------------------------

    columnas_finales = [
        "uid_infall",
        "uid_disruption",
        "uid_disruption_new",
        "motivo_disruption_cambia",
        "uids_asociadas_overlap",
        "snap_infall",
        "snap_disruption",
        "grupo",
        "motivo_conservacion",
        "flag_masa_modificada",
        "mass_star_total_infall",
        "mass_gas_total_infall",
        "mass_dm_total_infall",
        # >>> ADDED: resumen de fusiones post-infall por merger
        "n_post_infall",
        "uids_post_infall",
        "stellar_mass_ratio_post_infall",
        "baryonic_mass_ratio_post_infall",
        "uids_extra_infall", #filtro 8
        "uids_extra_infall_filtro5", #filtro 5
        "flag_conservado_noC2"
    ]

        
    df_conservados_final = df_conservados[columnas_finales].copy()
    df_conservados_final = df_conservados_final.sort_values(
        by="mass_star_total_infall", ascending=False
    ).reset_index(drop=True)
    



    # ============================================================
    # DESCRIPCIÓN DE COLUMNAS CLAVE – df_conservados_final
    # ============================================================

    # → uid_disruption_new:
    #     Se actualiza el tiempo de disrupción si el merger conservado vive más
    #     tiempo de lo que indica el árbol. Esto ocurre cuando otros mergers,
    #     previamente eliminados por overlap (filtros 1–3), en realidad formaban
    #     parte del mismo sistema, y uno de ellos sobrevive más tiempo.

    # → motivo_disruption_cambia:
    #     Indica qué tipo de solapamiento llevó al cambio de disrupción:
    #       - "inicioC2": (filtro 1) el merger más antiguo del grupo era tipo C2.
    #       - "contieneC2": (filtro 2) el merger absorbente no era el más antiguo,
    #         pero sí tipo C2.
    #       - "sinC": (filtro 3) no hay mergers tipo C, pero uno absorbe al resto.

    # → uids_asociadas_overlap:
    #     Lista de mergers eliminados por solapamiento estructural, cuyas trayectorias
    #     justificaron extender la disrupción del merger conservado.

    # → motivo_conservacion:
    #     Aparece solo si el grupo no es C2. Motivos típicos:
    #       - "rama rota": el merger pierde seguimiento en el árbol pero mantiene estructura.
    #       - "estructura preexistente": el merger fue separado por subfind, pero siempre
    #         formó parte de la galaxia central (e.g., particulado unbound que nunca
    #         fue subestructura real).

    # → flag_masa_modificada:
    #     True si la masa al infall fue corregida por:
    #       - Filtro 8 → `uids_extra_infall`: estructuras que se fusionan después del infall.
    #       - Filtro 5 → `uids_extra_infall_filtro5`: mergers A2 eliminados que suman masa
    #         al merger conservado.

    # → mass_*_total_infall:
    #     Masa total corregida al t_infall, incluyendo la masa original más los aportes
    #     estructurales ocultos detectados por los filtros.
    # ============================================================



    def obtener_particle_ids_uid(uid, path_hdf5):
        """Retorna los particle IDs de estrellas, gas y DM para un UID en su snapshot."""
        snap, subf = split_unique_id(int(uid))
        with h5py.File(path_hdf5, "r") as f:
            snap_grp = f[f"SnapNumber_{snap}"]

            offset_s = snap_grp["SubGroups/PartType4/Offsets"][subf].astype(int)
            offset_g = snap_grp["SubGroups/PartType0/Offsets"][subf].astype(int)
            offset_dm = snap_grp["SubGroups/PartType1/Offsets"][subf].astype(int)

            ids_s = snap_grp["PartType4/ParticleIDs"][offset_s[0]:offset_s[1]] if offset_s[1] > offset_s[0] else []
            ids_g = snap_grp["PartType0/ParticleIDs"][offset_g[0]:offset_g[1]] if offset_g[1] > offset_g[0] else []
            ids_dm = snap_grp["PartType1/ParticleIDs"][offset_dm[0]:offset_dm[1]] if offset_dm[1] > offset_dm[0] else []

            return list(ids_s), list(ids_g), list(ids_dm)

    # Crear tabla final con partículas por merger conservado
    rows = []

    for _, row in tqdm(df_conservados_final.iterrows(), total=len(df_conservados_final), desc="Extrayendo partículas"):

        uid = row["uid_infall"]
        uids_f5 = row["uids_extra_infall_filtro5"] or []
        uids_f8 = row["uids_extra_infall"] or []

        # --- Partículas merger principal ---
        d = dict_particulas_infall.get(str(uid), {})
        ids_star_main = d.get("stars", [])
        ids_gas_main = d.get("gas", [])
        ids_dm_main = d.get("dm", [])

        # --- Partículas filtro 5 (suma A2→C2) ---
        ids_star_f5, ids_gas_f5, ids_dm_f5 = [], [], []
        for uid_extra in uids_f5:
            try:
                s, g, dm = obtener_particle_ids_uid(uid_extra, path_hdf5)
                ids_star_f5.extend(s)
                ids_gas_f5.extend(g)
                ids_dm_f5.extend(dm)
            except Exception as e:
                print(f"⚠️ Error en filtro 5 ({uid_extra}): {e}")

        # --- Partículas filtro 8 (fusiones post-infall) ---
        ids_star_f8, ids_gas_f8, ids_dm_f8 = [], [], []
        for uid_extra in uids_f8:
            try:
                s, g, dm = obtener_particle_ids_uid(uid_extra, path_hdf5)
                ids_star_f8.extend(s)
                ids_gas_f8.extend(g)
                ids_dm_f8.extend(dm)
            except Exception as e:
                print(f"⚠️ Error en filtro 8 ({uid_extra}): {e}")

        # --- Total por tipo ---
        ids_star_total = list(ids_star_main) + list(ids_star_f5) + list(ids_star_f8)
        ids_gas_total  = list(ids_gas_main)  + list(ids_gas_f5)  + list(ids_gas_f8)
        ids_dm_total   = list(ids_dm_main)   + list(ids_dm_f5)   + list(ids_dm_f8)


        # --- Agregar fila ---
        rows.append({
            "uid_infall": uid,
            "particle_ids_star_main": list(ids_star_main),
            "particle_ids_star_f5": list(ids_star_f5),
            "particle_ids_star_f8": list(ids_star_f8),
            "particle_ids_star_total": list(ids_star_total),
            "particle_ids_gas_main": list(ids_gas_main),
            "particle_ids_gas_f5": list(ids_gas_f5),
            "particle_ids_gas_f8": list(ids_gas_f8),
            "particle_ids_gas_total": list(ids_gas_total),
            "particle_ids_dm_main": list(ids_dm_main),
            "particle_ids_dm_f5": list(ids_dm_f5),
            "particle_ids_dm_f8": list(ids_dm_f8),
            "particle_ids_dm_total": list(ids_dm_total)
        })

    # Convertir a DataFrame
    df_particulas_final = pd.DataFrame(rows)

    cols_de_listas = [col for col in df_particulas_final.columns if col.startswith("particle_ids_")]

    for col in cols_de_listas:
        df_particulas_final[col] = df_particulas_final[col].apply(safe_json_list)


    # >>> ADDED (opcional): tabla "exploded" por fusión post-infall
    
    # === Exportar detalle de fusiones post-infall (una fila por satélite) ===
    try:
        # Si no hay post-infall, no intento explotar la tabla
        keep = df_filtros_aplicados["n_post_infall"].fillna(0) > 0
        
        # --- asegurar snaps derivados de los uid_* antes del exploded ---
        def _uid_to_snap(u):
            # devuelve np.nan si u es NA o no se puede castear
            if pd.isna(u):
                return np.nan
            try:
                return split_unique_id(int(str(u)))[0]
            except Exception:
                return np.nan

        cols = df_filtros_aplicados.columns

        # snap_infall
        if "snap_infall" not in cols:
            df_filtros_aplicados["snap_infall"] = df_filtros_aplicados["uid_infall"].apply(_uid_to_snap)

        # elegir fuente para disruption: uid_disruption_new (si existe y no-NA) o uid_disruption
        if "uid_disruption_new" in cols:
            uid_disr_base = df_filtros_aplicados["uid_disruption_new"].where(
                df_filtros_aplicados["uid_disruption_new"].notna(),
                df_filtros_aplicados.get("uid_disruption")
            )
        else:
            uid_disr_base = df_filtros_aplicados["uid_disruption"]

        # snap_disruption
        if "snap_disruption" not in cols:
            df_filtros_aplicados["snap_disruption"] = uid_disr_base.apply(_uid_to_snap)


        if keep.any():
            base = df_filtros_aplicados.loc[keep, [
                "uid_infall", "uid_disruption", "snap_infall", "snap_disruption",
                "uids_post_infall", "stellar_mass_ratio_post_infall", "baryonic_mass_ratio_post_infall",
                "mass_star_total_infall", "mass_gas_total_infall"
            ]].copy()

            # Exploto columnas paralelas (uid y ratios) → una fila por satélite post-infall
            exploded = base.explode(
                ["uids_post_infall", "stellar_mass_ratio_post_infall", "baryonic_mass_ratio_post_infall"],
                ignore_index=True
            )

            # Quito filas vacías si hubiera alguna celda None tras explode
            exploded = exploded.dropna(subset=["uids_post_infall"])

            # Renombro columnas para claridad
            exploded.rename(columns={
                "uids_post_infall": "uid_satellite_infall",
                "stellar_mass_ratio_post_infall": "mr_star_post_infall",
                "baryonic_mass_ratio_post_infall": "mr_bary_post_infall",
            }, inplace=True)

            # Guardo con el mismo patrón de nombres que el resto de salidas
            exploded_path = f"{carpeta_out}/post_infall_events_{sim_name}_gx{target_subfind_id}.csv"
            exploded.to_csv(exploded_path, index=False)
            print(f"✅ Guardado detalle post-infall (exploded): {exploded_path}")
        else:
            print(f"ℹ️ No hay fusiones post-infall que exportar para {sim_name} gx={target_subfind_id}.")
    except Exception as e:
        print(f"[!] No se pudo exportar tabla exploded post-infall: {e}")

    # === Guardar SIEMPRE la tabla de partículas del infall (independiente del exploded) ===
    try:
        output_path = f"{carpeta_out}/particle_ids_infall_{sim_name}_gx{target_subfind_id}.csv"
        df_particulas_final.to_csv(output_path, index=False)
        print(f"✅ Guardado archivo de partículas: {output_path}")
    except Exception as e:
        print(f"[!] No se pudo guardar archivo de partículas: {e}")

    

    
    cols_de_listas = [col for col in df_conservados_final.columns if col.startswith("uids_extra_")]

    for col in cols_de_listas:
        df_conservados_final[col] = df_conservados_final[col].apply(safe_json_list)

    df_conservados_final.to_csv(f"{carpeta_out}/clean_mergers_{sim_name}_gx{target_subfind_id}.csv")
    df_filtros_aplicados.to_csv(f"{carpeta_out}/1ststep_mergers_{sim_name}_gx{target_subfind_id}.csv")




def parse_ids_string(val):
    """Parsea un string con IDs a lista[int]."""
    """Convierte string tipo '[123 456]' o '[123,456]' en lista de enteros."""
    if pd.isna(val):
        return []
    val = str(val).strip()
    if val.startswith("[") and val.endswith("]"):
        contenido = val[1:-1].strip()
        if not contenido:
            return []
        tokens = re.split(r"[,\s]+", contenido)
        return [int(t) for t in tokens if t.strip().isdigit()]
    return []



def clean_component(df, tipo):
    """Limpia solapamientos de IDs por componente entre fuentes main, f5 y f8."""
    cols = [f"particle_ids_{tipo}_{suf}" for suf in ["main", "f5", "f8"]]
    apariciones = defaultdict(list)

    # 1. Recolectar dónde aparece cada pid
    for _, row in df.iterrows():
        uid = row["uid_infall"]
        for col in cols:
            for pid in parse_ids_string(row[col]):
                apariciones[pid].append((uid, col))

    # 2. Decidir el dueño (menor uid_infall)
    dueño_por_pid = {
        pid: min([uid for uid, _ in ocurrencias])
        for pid, ocurrencias in apariciones.items()
    }

    # 3. Print detallado
    print(f"\n📋 RESUMEN DE DUPLICADOS ({tipo.upper()}):")
    for pid, ocurrencias in apariciones.items():
        if len(ocurrencias) > 1:
            dueño = dueño_por_pid[pid]
            print(f"\n🔂 ParticleID {pid}")
            print(f"   ✅ Se conserva en uid_infall={dueño}")
            eliminados = [(uid, col) for uid, col in ocurrencias if uid != dueño]
            if eliminados:
                print("   ❌ Eliminado de:")
                for uid, col in eliminados:
                    print(f"      - uid_infall={uid}, columna={col}")
            else:
                print("   🔎 ¡Extraño! No hubo duplicados fuera del dueño.")

    # 4. Limpiar
    nuevas_columnas = {
        f"{col}_clean": [] for col in cols
    }
    nuevas_columnas[f"{tipo}_duplicates_removed"] = []
    nuevas_columnas[f"{tipo}_row_cleaned"] = []

    for _, row in df.iterrows():
        uid = row["uid_infall"]
        removed = []
        row_cleaned = False

        for col in cols:
            ids = parse_ids_string(row[col])
            cleaned_ids = []
            for pid in ids:
                if dueño_por_pid.get(pid) == uid:
                    cleaned_ids.append(pid)
                else:
                    removed.append(pid)
                    row_cleaned = True
            nuevas_columnas[f"{col}_clean"].append(cleaned_ids)

        nuevas_columnas[f"{tipo}_duplicates_removed"].append(removed)
        nuevas_columnas[f"{tipo}_row_cleaned"].append(row_cleaned)

    for colname, values in nuevas_columnas.items():
        df[colname] = values

    return df



def obtener_masa_por_ids(path_hdf5, tipo, particle_ids, snap_num):
    """Suma masa de un conjunto de IDs leyendo el HDF5; incluye MassTable para DM."""
    """
    Devuelve la masa total de los particle_ids dados para un tipo (stars, gas, dm)
    en el snapshot indicado, usando emparejamiento directo ID → masa.
    """
    if len(particle_ids) == 0:
        return 0.0

    particle_ids = np.array(particle_ids, dtype=np.uint64)

    with h5py.File(path_hdf5, 'r') as f:
        snap_key = f"SnapNumber_{snap_num}"
        if snap_key not in f:
            print(f"⚠️ Snapshot {snap_key} no encontrado en el archivo HDF5.")
            return 0.0

        snap_grp = f[snap_key]
        h = snap_grp["Header/HubbleParam"][()]

        if tipo == "dm":
            mass_table = snap_grp["Header/MassTable"][()]
            return mass_table[1] * 1e10/h *  len(particle_ids)

        parttype = {"stars": "PartType4", "gas": "PartType0"}[tipo]

        try:
            ids_all = snap_grp[f"{parttype}/ParticleIDs"][:]
            masses_all = snap_grp[f"{parttype}/Masses"][:] * 1e10 / h
        except KeyError:
            return 0.0

        # Crear diccionario directo id → masa
        id_to_mass = dict(zip(ids_all, masses_all))

        # Sumar solo las masas de los IDs presentes
        return sum(id_to_mass.get(pid, 0.0) for pid in particle_ids)






def final_depuracion(sim_name, target_subfind_id, snap_target): 
    """Paso 5 (final): calcula masas al infall, supervivencia a z=0 y ratios globales; exporta CSVs finales."""

    # Paths
    path_hdf5 = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    carpeta_out = "Datos/Baryon_Mergers_Depurados/"
    path_df_particulas_infall = os.path.join(carpeta_out, f"particle_ids_infall_{sim_name}_gx{target_subfind_id}.csv")

    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"

    tree_1 = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)  # árbol hacia atrás
    tree_2 = nx.read_multiline_adjlist(path_tree)  # árbol hacia adelante

    uid_target = f"{snap_target}{target_subfind_id:06d}"
    
    os.makedirs(carpeta_out, exist_ok=True)

    def main_branch_completa(uid):
        """
        Devuelve la rama principal completa de un merger (hacia atrás y adelante).
        """
        subtree_fwd = nx.dfs_tree(tree_1, uid)
        rama_fwd = get_main_branch_unique_ids(subtree_fwd, uid)
        subtree_bwd = nx.dfs_tree(tree_2, uid)
        rama_bwd = get_main_branch_unique_ids(subtree_bwd, uid)[1:][::-1]
        return rama_bwd + rama_fwd

    def limpiar_main_branch(uids):
        """
        Recibe una lista de uids que representan una rama principal
        y devuelve un diccionario limpio con un único uid por snapshot.
        """
        uids = list(map(int, uids))
        dict_por_snap = {
            uid // 10**6: uid for uid in uids
        }
        return {snap: uid for snap, uid in sorted(dict_por_snap.items())}

    dict_rama_central = limpiar_main_branch(main_branch_completa(str(uid_target)))
    #df_particulas = pd.read_csv(path_df_particulas_infall)
    try:
        df_particulas = pd.read_csv(path_df_particulas_infall)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"[INFO] {path_df_particulas_infall} vacío o inexistente → exportando vacío con esquema estándar")
        columnas = [
            "uid_infall",
            "particle_ids_star_main", "particle_ids_star_f5", "particle_ids_star_f8",
            "particle_ids_gas_main",  "particle_ids_gas_f5",  "particle_ids_gas_f8",
            "particle_ids_dm_main",   "particle_ids_dm_f5",   "particle_ids_dm_f8",
        ]
        df_particulas = pd.DataFrame(columns=columnas)
        # salida rápida para no seguir procesando
        carpeta_out = "Datos/Baryon_Mergers_Depurados"
        os.makedirs(carpeta_out, exist_ok=True)
        out_csv = os.path.join(carpeta_out, f"baryon_mergers_dep_{sim_name}_gx_{target_subfind_id}.csv")
        df_particulas.to_csv(out_csv, index=False)
        return


    def obtener_uid_central_en_snap(uid_infall):
        """
        Para un merger dado (uid_infall), encuentra el UID de la galaxia central
        en el mismo snapshot.
        """

        snap_infall, _ = split_unique_id(int(uid_infall))
        return(dict_rama_central[snap_infall])


    # Copia base para trabajar
    df_cleaned = df_particulas.copy()

    # Aplicar a cada tipo
    for tipo in ["star", "gas", "dm"]:
        print(f"\n=== Procesando {tipo.upper()} ===")
        df_cleaned = clean_component(df_cleaned, tipo)

    # Agregar flags de modificación por tipo
    df_cleaned["star_modified"] = df_cleaned["star_row_cleaned"]
    df_cleaned["gas_modified"] = df_cleaned["gas_row_cleaned"]
    df_cleaned["dm_modified"] = df_cleaned["dm_row_cleaned"]

    # Opcional: ver solo filas modificadas en al menos una componente
    filas_modificadas = df_cleaned[df_cleaned[["star_modified", "gas_modified", "dm_modified"]].any(axis=1)]
    print(f"\n✅ Total de filas modificadas (en alguna componente): {len(filas_modificadas)}")

    for tipo in ["star", "gas", "dm"]:
        col_main = f"particle_ids_{tipo}_main_clean"
        col_f5   = f"particle_ids_{tipo}_f5_clean"
        col_f8   = f"particle_ids_{tipo}_f8_clean"
        col_total = f"particle_ids_{tipo}_infall_total"

        df_cleaned[col_total] = df_cleaned.apply(
            lambda row: (row.get(col_main) or []) + 
                        (row.get(col_f5) or []) + 
                        (row.get(col_f8) or []),
            axis=1
        )
    m_star, m_gas, m_dm = [], [], []

    print("\n🔄 Calculando masas reales al infall...")
    for _, row in tqdm(df_cleaned.iterrows(), total=len(df_cleaned), desc="Calculando masas"):

        uid = row["uid_infall"]
        snap, _ = split_unique_id(int(uid))

        ids_star = row.get("particle_ids_star_infall_total", [])
        ids_gas  = row.get("particle_ids_gas_infall_total", [])
        ids_dm   = row.get("particle_ids_dm_infall_total", [])

        m_star.append(obtener_masa_por_ids(path_hdf5, "stars", ids_star, snap))
        m_gas.append(obtener_masa_por_ids(path_hdf5, "gas", ids_gas, snap))
        m_dm.append(obtener_masa_por_ids(path_hdf5, "dm", ids_dm, snap))

    df_cleaned["mass_star_infall"] = m_star
    df_cleaned["mass_gas_infall"]  = m_gas
    df_cleaned["mass_dm_infall"]   = m_dm

    ID_SHIFT = 2**31

    # --- 1. Obtener IDs + masas en la galaxia central a z=0 ---

    with h5py.File(path_hdf5, 'r') as f:
        grp = f[f"SnapNumber_{snap_target}"]
        h = grp["Header/HubbleParam"][()]
        mass_table = grp["Header/MassTable"][()]

        # Offsets
        offsets = {
            "dm": grp["SubGroups/PartType1/Offsets"][()],
            "gas": grp["SubGroups/PartType0/Offsets"][()],
            "stars": grp["SubGroups/PartType4/Offsets"][()]
        }

        # PartType1 - DM (IDs + masa constante)
        ids_dm_all = grp["PartType1/ParticleIDs"][:]
        o_dm = offsets["dm"][target_subfind_id].astype('int')
        ids_dm_central = set(ids_dm_all[o_dm[0]:o_dm[1]])

        # PartType0 - Gas (IDs + masas)
        ids_gas_all = grp["PartType0/ParticleIDs"][:]
        masses_gas_all = grp["PartType0/Masses"][:] * 1e10 / h
        o_gas = offsets["gas"][target_subfind_id].astype('int')
        ids_gas_central = ids_gas_all[o_gas[0]:o_gas[1]]
        masses_gas_central = masses_gas_all[o_gas[0]:o_gas[1]]
        dict_gas_mass_z0 = dict(zip(ids_gas_central, masses_gas_central))

        # PartType4 - Stars (IDs + masas)
        ids_star_all = grp["PartType4/ParticleIDs"][:]
        masses_star_all = grp["PartType4/Masses"][:] * 1e10 / h
        o_star = offsets["stars"][target_subfind_id].astype('int')
        ids_star_central = ids_star_all[o_star[0]:o_star[1]]
        masses_star_central = masses_star_all[o_star[0]:o_star[1]]
        dict_star_mass_z0 = dict(zip(ids_star_central, masses_star_central))

    # --- 2. Calcular masas sobrevivientes y formación estelar ---

    m_star_z0, m_gas_z0, m_dm_z0 = [], [], []
    m_star_from_gas_1gen, m_star_from_gas_2gen, m_star_from_gas_2gen_unique = [], [], []

    print("\n🔄 Calculando masas sobrevivientes y formación estelar...")
    for _, row in tqdm(df_cleaned.iterrows(), total=len(df_cleaned)):

        ids_star = row.get("particle_ids_star_infall_total", [])
        ids_gas  = row.get("particle_ids_gas_infall_total", [])
        ids_dm   = row.get("particle_ids_dm_infall_total", [])

        # Estrellas sobrevivientes
        m_star_z0.append(sum(dict_star_mass_z0.get(pid, 0.0) for pid in ids_star))

        # Gas sobreviviente
        m_gas_z0.append(sum(dict_gas_mass_z0.get(pid, 0.0) for pid in ids_gas))

        # DM sobreviviente (masa fija)
        num_dm = sum(1 for pid in ids_dm if pid in ids_dm_central)
        m_dm_z0.append(num_dm * mass_table[1] * 1e10 / h)

        # Formación estelar a partir de gas
        m1, m2, m2u = 0.0, 0.0, 0.0
        for pid in ids_gas:
            if pid < ID_SHIFT:
                if pid in dict_star_mass_z0:
                    m1 += dict_star_mass_z0[pid]
                if (pid + ID_SHIFT) in dict_star_mass_z0:
                    m2 += dict_star_mass_z0[pid + ID_SHIFT]
            else:
                if pid in dict_star_mass_z0:
                    m2u += dict_star_mass_z0[pid]
        m_star_from_gas_1gen.append(m1)
        m_star_from_gas_2gen.append(m2)
        m_star_from_gas_2gen_unique.append(m2u)

    # --- 3. Agregar columnas al DataFrame ---

    df_cleaned["mass_star_z0_central"] = m_star_z0
    df_cleaned["mass_gas_z0_central"]  = m_gas_z0
    df_cleaned["mass_dm_z0_central"]   = m_dm_z0

    df_cleaned["mass_star_from_gas_1gen_z0_central"]        = m_star_from_gas_1gen
    df_cleaned["mass_star_from_gas_2gen_z0_central"]        = m_star_from_gas_2gen
    df_cleaned["mass_star_from_gas_2gen_unique_z0_central"] = m_star_from_gas_2gen_unique

    print("\n✅ Columnas añadidas correctamente con masas a z=0.")

    # --- Inicializar listas para almacenar IDs que quedan en la central ---
    ids_star_z0, ids_gas_z0, ids_dm_z0 = [], [], []
    ids_star_from_gas_1gen, ids_star_from_gas_2gen, ids_star_from_gas_2gen_unique = [], [], []

    print("\n🔄 Guardando IDs que quedan en la galaxia central a z=0...")
    for _, row in tqdm(df_cleaned.iterrows(), total=len(df_cleaned)):

        ids_star = row.get("particle_ids_star_infall_total", [])
        ids_gas  = row.get("particle_ids_gas_infall_total", [])
        ids_dm   = row.get("particle_ids_dm_infall_total", [])

        # === Stars que siguen en la central ===
        ids_star_z0.append([pid for pid in ids_star if pid in dict_star_mass_z0])

        # === Gas que sigue en la central ===
        ids_gas_z0.append([pid for pid in ids_gas if pid in dict_gas_mass_z0])

        # === DM que sigue en la central ===
        ids_dm_z0.append([pid for pid in ids_dm if pid in ids_dm_central])

        # === Estrellas formadas desde gas ===
        ids_1gen, ids_2gen, ids_2gen_unique = [], [], []

        for pid in ids_gas:
            if pid < ID_SHIFT:
                if pid in dict_star_mass_z0:
                    ids_1gen.append(pid)
                if (pid + ID_SHIFT) in dict_star_mass_z0:
                    ids_2gen.append(pid + ID_SHIFT)
            else:
                if pid in dict_star_mass_z0:
                    ids_2gen_unique.append(pid)

        ids_star_from_gas_1gen.append(ids_1gen)
        ids_star_from_gas_2gen.append(ids_2gen)
        ids_star_from_gas_2gen_unique.append(ids_2gen_unique)

    # --- Agregar columnas al DataFrame ---
    df_cleaned["ids_star_z0_central"] = ids_star_z0
    df_cleaned["ids_gas_z0_central"]  = ids_gas_z0
    df_cleaned["ids_dm_z0_central"]   = ids_dm_z0

    df_cleaned["ids_star_from_gas_1gen_z0_central"]        = ids_star_from_gas_1gen
    df_cleaned["ids_star_from_gas_2gen_z0_central"]        = ids_star_from_gas_2gen
    df_cleaned["ids_star_from_gas_2gen_unique_z0_central"] = ids_star_from_gas_2gen_unique

    print("\n✅ Columnas con IDs añadidas a df_cleaned.")

    mass_star_central_infall = []
    mass_gas_central_infall = []
    mass_dm_central_infall = []

    for _, row in tqdm(df_cleaned.iterrows(), total=len(df_cleaned), desc="Cargando masa central en infall"):

        uid_infall = int(row["uid_infall"])
        snap, _ = split_unique_id(uid_infall)
        uid_central = obtener_uid_central_en_snap(str(uid_infall))

        if uid_central is None:
            mass_star_central_infall.append(np.nan)
            mass_gas_central_infall.append(np.nan)
            mass_dm_central_infall.append(np.nan)
            continue

        snap_infall, subfind_id = split_unique_id(uid_central)

        # Calcular masas en ese subhalo
        with h5py.File(path_hdf5, 'r') as f:
            grp = f[f"SnapNumber_{snap_infall}"]
            h = grp["Header/HubbleParam"][()]
            mass_table = grp["Header/MassTable"][()]

            # --- Stars
            o_star = grp["SubGroups/PartType4/Offsets"][subfind_id].astype(int)
            m_star = np.sum(grp["PartType4/Masses"][o_star[0]:o_star[1]]) * 1e10 / h
            mass_star_central_infall.append(m_star)

            # --- Gas
            o_gas = grp["SubGroups/PartType0/Offsets"][subfind_id].astype(int)
            m_gas = np.sum(grp["PartType0/Masses"][o_gas[0]:o_gas[1]]) * 1e10 / h
            mass_gas_central_infall.append(m_gas)

            # --- DM
            o_dm = grp["SubGroups/PartType1/Offsets"][subfind_id].astype(int)
            n_dm = o_dm[1] - o_dm[0]
            m_dm = n_dm * mass_table[1] * 1e10 / h
            mass_dm_central_infall.append(m_dm)

    df_cleaned["mass_star_central_infall"] = mass_star_central_infall
    df_cleaned["mass_gas_central_infall"]  = mass_gas_central_infall
    df_cleaned["mass_dm_central_infall"]   = mass_dm_central_infall

    df_cleaned["stellar_mass_ratio"] = df_cleaned["mass_star_infall"] / df_cleaned["mass_star_central_infall"]
    df_cleaned["baryonic_mass_ratio"] = (
        (df_cleaned["mass_star_infall"] + df_cleaned["mass_gas_infall"]) /
        (df_cleaned["mass_star_central_infall"] + df_cleaned["mass_gas_central_infall"])
    )

    # uid_infall : ID único del halo en el momento del infall.
    # particle_ids_star_main : IDs de partículas estelares en el halo principal.
    # particle_ids_star_f5 : Estrellas capturadas por el filtro 5 (fusiones A2→C2).
    # particle_ids_star_f8 : Estrellas provenientes de fusiones post-infall (filtro 8).
    # particle_ids_star_total : Unión directa de main + f5 + f8 (sin eliminar duplicados).

    # particle_ids_gas_main : IDs de partículas de gas en el halo principal.
    # particle_ids_gas_f5 : Gas capturado por el filtro 5.
    # particle_ids_gas_f8 : Gas proveniente de fusiones post-infall.
    # particle_ids_gas_total : Unión directa de main + f5 + f8 (sin eliminar duplicados).

    # particle_ids_dm_main : IDs de partículas de materia oscura (DM) en el halo principal.
    # particle_ids_dm_f5 : DM capturada por el filtro 5.
    # particle_ids_dm_f8 : DM de fusiones post-infall.
    # particle_ids_dm_total : Unión directa de main + f5 + f8 (sin eliminar duplicados).

    # particle_ids_star_main_clean : Versión limpia (sin duplicados) de estrellas del halo principal.
    # particle_ids_star_f5_clean : Versión limpia de estrellas de f5.
    # particle_ids_star_f8_clean : Versión limpia de estrellas de f8.
    # star_duplicates_removed : Lista de particleIDs de estrellas eliminadas por estar duplicadas.
    # star_row_cleaned : True si se removieron duplicados en alguna columna estelar.

    # particle_ids_gas_main_clean : Gas main limpio.
    # particle_ids_gas_f5_clean : Gas f5 limpio.
    # particle_ids_gas_f8_clean : Gas f8 limpio.
    # gas_duplicates_removed : IDs de gas eliminados por duplicación.
    # gas_row_cleaned : True si hubo limpieza de duplicados en columnas de gas.

    # particle_ids_dm_main_clean : DM main limpio.
    # particle_ids_dm_f5_clean : DM f5 limpio.
    # particle_ids_dm_f8_clean : DM f8 limpio.
    # dm_duplicates_removed : IDs de DM eliminados por duplicación.
    # dm_row_cleaned : True si hubo limpieza de duplicados en columnas de DM.

    # star_modified, gas_modified, dm_modified : Flags generales si se modificó algo en cada componente.
    ######################################################
    # particle_ids_star_infall_total : Unión limpia (main_clean + f5_clean + f8_clean) de estrellas al infall.
    # particle_ids_gas_infall_total : Unión limpia de gas al infall.
    # particle_ids_dm_infall_total : Unión limpia de DM al infall.
    ######################################################
    # mass_star_infall : Masa total estelar al infall, considerando los particleids limpios
    # mass_gas_infall : Masa total de gas al infall.
    # mass_dm_infall : Masa de materia oscura al infall.
    ######################################################
    # mass_star_z0_central : Masa estelar a z=0 aún presente en la galaxia central, considerando los particleids limpios
    # mass_gas_z0_central : Masa de gas a z=0 en la galaxia central.
    # mass_dm_z0_central : Masa de DM a z=0 en la galaxia central.

    # mass_star_from_gas_1gen_z0_central : Masa estelar en z=0 de estrellas formadas a partir de gas con el mismo ID.
    # mass_star_from_gas_2gen_z0_central : Masa estelar de estrellas formadas con ID = gas + 2**31.
    # mass_star_from_gas_2gen_unique_z0_central : Estrellas formadas con ID > 2**31 directamente presentes a z=0.
    ######################################################
    # ids_star_z0_central : ParticleIDs estelares sobrevivientes en la galaxia central a z=0.
    # ids_gas_z0_central : ParticleIDs de gas aún presentes en la galaxia central.
    # ids_dm_z0_central : IDs de DM presentes en la galaxia central a z=0.

    # ids_star_from_gas_1gen_z0_central : IDs de estrellas de 1ra generación formadas desde gas.
    # ids_star_from_gas_2gen_z0_central : IDs de 2da generación con offset 2**31.
    # ids_star_from_gas_2gen_unique_z0_central : IDs > 2**31 únicos a z=0.
    ######################################################
    # mass_star_central_infall : Masa estelar de la galaxia central (en ese snapshot de infall).
    # mass_gas_central_infall : Masa de gas de la central al snapshot del infall.
    # mass_dm_central_infall : Masa de DM de la central al momento del infall.

    # stellar_mass_ratio : Cociente entre masa estelar del merger y de la galaxia central en el snapshot de infall.
    # baryonic_mass_ratio : Idem anterior, pero usando masa gas + estrellas.


    def safe_json_list(val):
        if isinstance(val, list):
            return json.dumps(val)
        return "[]"

    # Detectar automáticamente columnas con listas
    cols_de_listas = [
        col for col in df_cleaned.columns
        if df_cleaned[col].apply(lambda x: isinstance(x, list)).any()
    ]

    # Serializar a JSON
    for col in cols_de_listas:
        df_cleaned[col] = df_cleaned[col].apply(safe_json_list)

    # Guardar
    output_path = f"{carpeta_out}/FINAL_clean_mergers_{sim_name}_gx{target_subfind_id}.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"✅ Guardado: {output_path}")

    # 🔄 Selección de columnas clave para análisis resumido
    columnas_resumen = [
        "uid_infall",  # ID único del merger (codifica snapshot y subhalo)

        # --- Masas al infall (en el snapshot correspondiente al merger) ---
        "mass_star_infall",  # Masa total estelar al infall
        "mass_gas_infall",   # Masa total de gas al infall
        "mass_dm_infall",    # Masa total de materia oscura al infall

        # --- Masas a z=0 en la galaxia central objetivo ---
        "mass_star_z0_central",  # Masa estelar que sobrevive en la central a z=0
        "mass_gas_z0_central",   # Masa de gas que sobrevive a z=0
        "mass_dm_z0_central",    # Masa de DM que sobrevive a z=0

        # --- Formación estelar a partir del gas (presente a z=0) ---
        "mass_star_from_gas_1gen_z0_central",        # Masa estelar en z=0 de estrellas formadas a partir de gas con el mismo ID
        "mass_star_from_gas_2gen_z0_central",        # Masa estelar de estrellas formadas con ID = gas + 2**31
        "mass_star_from_gas_2gen_unique_z0_central", # Estrellas formadas con ID > 2**31 directamente presentes a z=0

        # --- Masas de la galaxia central en el snapshot del infall ---
        "mass_star_central_infall",  # Masa estelar de la central en el snapshot del merger
        "mass_gas_central_infall",   # Masa de gas de la central al momento del merger
        "mass_dm_central_infall",    # Masa de DM de la central en ese snapshot

        # --- Razones de masa (merger vs. central en el infall) ---
        "stellar_mass_ratio",   # Ratio de masa estelar merger / central
        "baryonic_mass_ratio"   # Ratio de (masa estelar + gas) merger / central
    ]

    # 💾 Extraer resumen
    df_resumen = df_cleaned[columnas_resumen].copy()

    output_path = f"{carpeta_out}/FINAL_resumen_clean_mergers_{sim_name}_gx{target_subfind_id}.csv"
    df_resumen.to_csv(output_path, index=False)
    print(f"✅ Guardado resumen: {output_path}")

    # 🔍 Verificar duplicados internos en las listas _infall_total

    def revisar_duplicados_en_lista(col):
        print(f"\n🔎 Verificando duplicados en: {col}")
        duplicados = []

        for idx, ids in df_cleaned[col].items():
            if not isinstance(ids, list):
                continue  # saltar si no es lista
            ids_set = set(ids)
            if len(ids) != len(ids_set):
                duplicados.append(idx)

        if duplicados:
            print(f"❗ Duplicados encontrados en {len(duplicados)} filas: {duplicados[:10]}{'...' if len(duplicados) > 10 else ''}")
        else:
            print("✅ Sin duplicados internos en esta columna.")

    # Aplicar a todas las columnas _infall_total
    for col in [
        "particle_ids_star_infall_total",
        "particle_ids_gas_infall_total",
        "particle_ids_dm_infall_total"
    ]:
        revisar_duplicados_en_lista(col)


    # === Parameters ===
    gxid = target_subfind_id
    snap = snap_target

    # === Paths ===
    base_tabla = f"Datos/Tablitas/{sim_name}/data_{gxid}"
    path_stars = os.path.join(base_tabla, "info_galaxy", f"{snap}_info.parquet")
    path_gas   = os.path.join(base_tabla, "info_gas", f"{snap}_infogas.parquet")
    path_df    = f"Datos/Baryon_Mergers_Depurados/FINAL_clean_mergers_{sim_name}_gx{gxid}.csv"


    # === Load data ===
    df_stars = pd.read_parquet(path_stars)
    df_info  = pd.read_csv(path_df)
    has_gas  = os.path.exists(path_gas)

    df_gas = pd.read_parquet(path_gas) if has_gas else None

    # === Safe list parsing ===
    def safe_list(val):
        if pd.isna(val) or val == "":
            return []
        return literal_eval(val)

    cols_ids = [
        "ids_star_from_gas_1gen_z0_central",
        "ids_star_from_gas_2gen_z0_central",
        "ids_star_from_gas_2gen_unique_z0_central",
        "particle_ids_star_infall_total",
        "particle_ids_gas_infall_total"
    ]
    for col in cols_ids:
        df_info[col] = df_info[col].apply(safe_list)

    # === Unions ===
    ids_1gen         = set().union(*df_info["ids_star_from_gas_1gen_z0_central"])
    ids_2gen         = set().union(*df_info["ids_star_from_gas_2gen_z0_central"])
    ids_2uniq        = set().union(*df_info["ids_star_from_gas_2gen_unique_z0_central"])
    ids_star_mergers = set().union(*df_info["particle_ids_star_infall_total"])
    ids_gas_mergers  = set().union(*df_info["particle_ids_gas_infall_total"])

    # === Filtrar partículas ===
    stars_1gen     = df_stars[df_stars["ids"].isin(ids_1gen)]
    stars_2gen     = df_stars[df_stars["ids"].isin(ids_2gen)]
    stars_2uniq    = df_stars[df_stars["ids"].isin(ids_2uniq)]
    stars_mergers  = df_stars[df_stars["ids"].isin(ids_star_mergers)]
    gas_mergers    = df_gas[df_gas["ids"].isin(ids_gas_mergers)] if has_gas else pd.DataFrame()

    # === Plot ===
    alpha_val = 0.2
    s_val = 0.1
    legend_style = dict(markerscale=15, scatterpoints=1, loc="upper right", frameon=False)

    if not gas_mergers.empty:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
    else:
        fig, axs_raw = plt.subplots(1, 2, figsize=(12, 5))
        axs = [axs_raw[0], axs_raw[1]]

    # Stars XY
    axs[0].scatter(stars_mergers["x"], stars_mergers["y"], s=s_val, color="gray", alpha=alpha_val, label="Mergers (stars)")
    axs[0].scatter(stars_1gen["x"], stars_1gen["y"], s=s_val, color="red", alpha=alpha_val, label="Formed (1st gen)")
    axs[0].scatter(stars_2gen["x"], stars_2gen["y"], s=s_val, color="gold", alpha=alpha_val, label="Formed (2nd gen)")
    axs[0].scatter(stars_2uniq["x"], stars_2uniq["y"], s=s_val, color="blue", alpha=alpha_val, label="Formed (2nd gen uniq)")
    axs[0].set_title("Stars: $x$ vs $y$")
    axs[0].set_xlabel("$x$ [kpc]")
    axs[0].set_ylabel("$y$ [kpc]")
    axs[0].set_aspect('equal')
    axs[0].legend(**legend_style)

    # Stars XZ
    axs[1].scatter(stars_mergers["x"], stars_mergers["z"], s=s_val, color="gray", alpha=alpha_val, label="Mergers (stars)")
    axs[1].scatter(stars_1gen["x"], stars_1gen["z"], s=s_val, color="red", alpha=alpha_val, label="Formed (1st gen)")
    axs[1].scatter(stars_2gen["x"], stars_2gen["z"], s=s_val, color="gold", alpha=alpha_val, label="Formed (2nd gen)")
    axs[1].scatter(stars_2uniq["x"], stars_2uniq["z"], s=s_val, color="blue", alpha=alpha_val, label="Formed (2nd gen uniq)")
    axs[1].set_title("Stars: $x$ vs $z$")
    axs[1].set_xlabel("$x$ [kpc]")
    axs[1].set_ylabel("$z$ [kpc]")
    axs[1].set_aspect('equal')
    axs[1].legend(**legend_style)

    # Optional gas panels
    if not gas_mergers.empty:
        axs[2].scatter(gas_mergers["x"], gas_mergers["y"], s=s_val, color="teal", alpha=alpha_val, label="Mergers (gas)")
        axs[2].set_title("Gas: $x$ vs $y$")
        axs[2].set_xlabel("$x$ [kpc]")
        axs[2].set_ylabel("$y$ [kpc]")
        axs[2].set_aspect('equal')
        axs[2].legend(**legend_style)

        axs[3].scatter(gas_mergers["x"], gas_mergers["z"], s=s_val, color="teal", alpha=alpha_val, label="Mergers (gas)")
        axs[3].set_title("Gas: $x$ vs $z$")
        axs[3].set_xlabel("$x$ [kpc]")
        axs[3].set_ylabel("$z$ [kpc]")
        axs[3].set_aspect('equal')
        axs[3].legend(**legend_style)

    fig.suptitle(f'{sim_name} - Galaxy {gxid}', fontsize=14)
    plt.tight_layout()
    plt.show()



# ============================ PIPELINE WRAPPER ===============================
def run_merger_pipeline(sim_name: str, target_subfind_id: int, snap_target: int):
    """
    Ejecuta el pipeline completo para UNA galaxia (STEP 4 + STEP 5) sin cambiar lógica.
    """
    carpeta_out = "Datos/Baryon_Mergers_Depurados/"
    path_guardado = f"{carpeta_out}/FINAL_resumen_clean_mergers_{sim_name}_gx{target_subfind_id}.csv"
    if os.path.exists(path_guardado):
        print(f'Me salto {sim_name}-{target_subfind_id} porque ya fue procesado')
        # Ya procesado: continuar al siguiente snapshot
        return
    depura_mergers_trees(sim_name, target_subfind_id, snap_target)
    final_depuracion(sim_name, target_subfind_id, snap_target)

depura_mergers_trees_full = run_merger_pipeline


# ================================ MAIN ===================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Depuración + Finalización de mergers (unificado)")
    ap.add_argument("--sim", required=True)
    ap.add_argument("--gx", type=int, required=True)
    ap.add_argument("--snap0", type=int, required=True)
    args = ap.parse_args()
    run_merger_pipeline(args.sim, args.gx, args.snap0)

if __name__ == "__main__":
    main()
