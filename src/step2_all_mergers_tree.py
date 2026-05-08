#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Catalina Casanueva (cicasanueva@uc.cl), 07/2025
# ============================================================
# Paso [2] – Detección de mergers usando tree (de cualquier tipo: sólo DM, bariónicos) 
# para una galaxia central dada (identificada por su subfind ID en z = 0).
#
# Estructura de entrada:
# ----------------------
# - Árbol de mergers generado con el código AMIGA (Knollmann, S. R. & Knebe, A. 2009, ApJS, 182, 608).
# - Archivo HDF5 con los snapshots de la simulación CIELO.
#
# Procedimiento:
# --------------
# 1. Se extrae la *main branch completa* de la galaxia central (`uid_target`).
# 2. Se identifican todas las ramas que terminan fusionándose físicamente con esa main branch:
#    - Se recorre el grafo invertido (`G_inv`) desde `uid_target` hacia atrás.
#    - Se excluyen:
#      · Las ramas que ya pertenecen a la main branch.
#      · Las ramas que se fusionaron previamente con otra que sí lo hace.
#
#    Por ejemplo:
#       Supongamos que A y B son galaxias progenitoras y en snapshot 60 ocurre A + B → C,
#       y luego en snapshot 50, C se fusiona con la galaxia central.
#       En este caso, solo se conserva la rama de A con A más masiva que B.
#
#    Esto evita contar múltiples veces el mismo evento de fusión física.
#
# 3. Para cada merger físico detectado, se reconstruye su rama principal:
#    - Desde el último snapshot antes de fusionarse con la central (`uid_before_merge`, i.e. último uid
#      que tenga distinto a la central)
#    - Se recorre hacia atrás (progenitores) y hacia adelante (descendientes) combinando `G` y `G_fwd`.
#
# Archivo de salida:
# -------------------
# (1) CSV resumen:
#     - `uid_before_merge`: último UID antes de tocar la main branch de la central.
#     - `uid_merge`: UID donde se fusiona con la central (coincide con la rama principal).
#     - `snap_merge`: snapshot de la fusión.
#     - `z_merge`: redshift correspondiente.
#     - `origin_node`: nodo raíz de la galaxia secundaria.
# ============================================================

import os
import h5py
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import split_unique_id, get_main_branch_unique_ids

# ------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------

def all_mergers_tree(sim_name, target_subfind_id, snap_target):
    #sim_name = "LG1"
    #target_subfind_id = 4337
    #snap_target = 128
    uid_target = f"{snap_target}{target_subfind_id:06d}"

    # Paths de entrada y salida
    path_sim = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"
    
    carpeta_out = "Datos/All_Mergers/"
    
    # Crea la carpeta si no existe
    os.makedirs(carpeta_out, exist_ok=True)   
    
    #parquet_out = f"{carpeta_out}/merging_secondary_mainbranches_{sim_name}_gx_{target_subfind_id}.parquet"
    csv_out = f"{carpeta_out}/mergers_{sim_name}_gx_{target_subfind_id}_summary.csv"
    
    if os.path.exists(csv_out):
        print(f'Me salto {sim_name}-{target_subfind_id} porque ya fue procesado')
        # Ya procesado: continuar al siguiente snapshot
        return


    # ------------------------------------------------------------
    # Carga de redshift por snapshot
    # ------------------------------------------------------------
    z_dict = {}
    with h5py.File(path_sim, 'r') as f:
        for k in f.keys():
            if k.startswith("SnapNumber_"):
                snap = int(k.split("_")[1])
                try:
                    z = f[k]["Header"]["Redshift"][()]
                    z_dict[snap] = z
                except:
                    continue
    all_snaps = sorted(z_dict)

    # ------------------------------------------------------------
    # Carga de árboles y main branch
    # ------------------------------------------------------------
    print(f"[INFO] Cargando árboles y main branch desde UID {uid_target}")
    G = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)  # árbol hacia atrás
    G_fwd = nx.read_multiline_adjlist(path_tree)                       # árbol hacia adelante
    G_inv = G.reverse(copy=True)

    # Extraer main branch completa de la central
    subtree_main = nx.dfs_tree(G, uid_target)
    main_branch_ids = set(get_main_branch_unique_ids(subtree_main, uid_target))

    # ------------------------------------------------------------
    # Detección de mergers por tree
    # ------------------------------------------------------------
    print(f"[INFO] Detectando por tree")
    merging_branches = {}

    # Usamos shortest_path desde todos los nodos hasta el central
    shortest_paths = nx.single_target_shortest_path(G_inv, uid_target)

    for node, path in tqdm(shortest_paths.items()):
        if node == uid_target or node in main_branch_ids:
            continue
        if len(path) < 2:
            continue

        for i, uid in enumerate(path):
            if uid in main_branch_ids:
                idx_merger = i
                break
        else:
            continue  # no hay fusión real

        uid_before = path[idx_merger - 1]

        # Verificamos que esta rama realmente se fusiona y no es parte previa de la main
        if any(uid in main_branch_ids for uid in get_main_branch_unique_ids(nx.dfs_tree(G, node), node)):
            continue

        # Ramas hacia atrás y adelante desde el punto de fusión
        branch_back = get_main_branch_unique_ids(nx.dfs_tree(G, uid_before), uid_before)
        try:
            branch_forward = get_main_branch_unique_ids(nx.dfs_tree(G_fwd, uid_before), uid_before)
        except:
            branch_forward = []

        # Concatenar sin repetir UIDs
        full_branch = list(dict.fromkeys(branch_back + branch_forward))
        merging_branches[uid_before] = full_branch

    # ------------------------------------------------------------
    # Construcción de tablas: CSV resumen
    # ------------------------------------------------------------
    #rows_parquet = []
    rows_csv = []

    for uid_merge, branch in merging_branches.items():
        uid_by_snap = {split_unique_id(int(uid))[0]: uid for uid in branch}
        #row = {"origin_node": uid_merge}

        #for snap in all_snaps:
        #    row[f"snap_{snap:03d}"] = uid_by_snap.get(snap, "")
        #rows_parquet.append(row)

        # Detectar momento de fusión
        snaps = sorted(uid_by_snap)
        try:
            snap_merge = min(s for s in snaps if uid_by_snap[s] in main_branch_ids)
            snap_prev = max(s for s in snaps if s < snap_merge)
            rows_csv.append({
                "origin_node": uid_merge,
                "snap_merge": snap_merge,
                "z_merge": z_dict.get(snap_merge, None),
                "uid_merge": uid_by_snap[snap_merge],
                "uid_before_merge": uid_by_snap[snap_prev],
            })
        except:
            rows_csv.append({
                "origin_node": uid_merge,
                "snap_merge": None,
                "z_merge": None,
                "uid_merge": None,
                "uid_before_merge": None,
            })

    # ------------------------------------------------------------
    # Guardado de resultados
    # ------------------------------------------------------------
    #df_parquet = pd.DataFrame(rows_parquet)
    #df_parquet.to_parquet(parquet_out, index=False)
    #print(f"[✅] Guardado Parquet en {parquet_out} con {len(df_parquet)} ramas fusionadas")

    if rows_csv:
        df_csv = pd.DataFrame(rows_csv)
        df_csv.to_csv(csv_out, index=False)
        print(f"[✅] Guardado CSV resumen en {csv_out}")
    else:
        df_csv = pd.DataFrame(columns=[
            "origin_node",
            "snap_merge",
            "z_merge",
            "uid_merge",
            "uid_before_merge"
        ])
        df_csv.to_csv(csv_out, index=False)
        print(f"[⚠️] Guardado CSV vacío en {csv_out} (sin mergers detectados para {sim_name} - {target_subfind_id})")


    # ------------------------------------------------------------
    # PRINT OPCIONAL: Estadísticas y resumen final
    # ------------------------------------------------------------
    print("\n" + "="*40)
    print(f"\nEstadísticas del resumen de mergers para {sim_name} - {target_subfind_id} (Paso [2])")
    print("\n" + "="*40)
    
    print(f"   🔹 Total de mergers físicos detectados: {len(df_csv)}")

    num_ok = df_csv["snap_merge"].notna().sum()
    num_bad = df_csv["snap_merge"].isna().sum()

    print(f"   🔹 Con 'snap_merge' bien definido: {num_ok}")
    print(f"   🔹 Con 'snap_merge' indefinido (problemas de tracking): {num_bad}")

    # Distribución por snapshot de fusión
    if num_ok > 0:
        print("\n   🔹 Distribución de mergers por snapshot:")
        distrib = df_csv["snap_merge"].value_counts().sort_index().reset_index()
        distrib.columns = ["Snapshot", "N_mergers"]
        print(distrib.to_string(index=False))

