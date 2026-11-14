# Room_scan : scanner une pièce au LiDAR et en faire un domaine de Physical_engine

Deux morceaux indépendants :

| Dossier / fichier | Rôle |
|---|---|
| `ios/` | application iPhone **RoomScan** (LiDAR requis) : maillage temps réel ARKit, export du dossier brut et du maillage (`mesh.ply` binaire, classe par face), mémoire bornée pour un appartement entier, journal `capture_log.txt`. Compilation sur Mac : voir `ios/README_mac.md` |
| `scan_to_domain.py` | PC : dossier `keyframes/` (RoomScan, Polycam) ou prise SpatialCapture → `solids.npy` + cas JSON de Physical_engine ; plusieurs scans recalés → un seul domaine |
| `register_scans.py` | PC : recalage global de scans faits à des moments différents dans le repère du premier (`alignment.json`), a priori sol + murs puis ICP Open3D |
| `compare_scans.py` | PC : différences entre deux versions d'un domaine (apparu / disparu), jumeau numérique |
| `detect_openings.py` | PC : portes et fenêtres reconnues par ARKit sur `mesh.ply` → rectangles de mailles (`"openings"` du cas JSON), proposés dans le moteur. Appelé par `scan_to_domain.py` ; seul, il complète un domaine déjà converti |
| `make_test_apartment.py` | masque synthétique d'appartement utilisé par les tests du moteur |
| `tests/test_format.py` | pièce synthétique tracée analytiquement au format RoomScan, vérifiée après conversion |
| `tests/test_spatialcapture.py` | la même pièce au format SpatialCapture (images répétées, suivi limité, NaN, ligne coupée) |
| `tests/test_register.py` | la pièce scannée deux fois dans deux repères : transformation retrouvée, domaine commun |
| `tests/test_compare.py` | la pièce puis la pièce avec une armoire : l'armoire est « apparue » |

## Chaîne complète

1. iPhone : RoomScan > Nouveau scan, parcourir lentement la pièce (sol, murs, plafond), Terminer, Partager.
2. PC :
   ```bat
   python scan_to_domain.py <scan>\keyframes --out <dossier> --name salon --dx 0.05
   ```
   Sorties : `solids.npy` (0 air, 1 solide), `colors.npy` (couleur des mailles solides, tirée des photos), `salon.json`,
   `apercu_*.png`, `voxels.ply` (coloré), `rapport.txt`.
3. Moteur :
   ```bat
   physical --case <dossier>\salon.json --run
   ```
   Le cas généré choisit le modèle d'air anélastique du moteur (pas de temps ~dx / 1 m/s : une pièce de 250 000 mailles
   de 5 cm tourne à ~10 fois le temps réel).

Le Python de Miniconda (numpy, Pillow, scipy) suffit : aucune installation. Seul le recalage fin de plusieurs scans
(`register_scans.py`) demande en plus Open3D (`pip install open3d`, ~100 Mo) ; sans lui il s'arrête à l'étape grossière.

## Un lieu, plusieurs scans

Trois façons de couvrir un appartement en plusieurs fois, et une pour suivre ses changements :

1. **Pause** dans l'app : la session ARKit continue, on reprend là où on était (même repère, rien à faire sur PC).
2. **Reprendre le scan** (fiche du scan dans l'app) : la carte ARKit sauvegardée à la fin du scan
   (`worldmap.arworldmap`) est rechargée, ARKit se relocalise dès qu'il reconnaît un endroit déjà scanné, et les
   nouvelles images clés, le nouveau maillage et les zones manquantes s'ajoutent **dans le même repère**. Le dossier
   reste un seul scan : `scan_to_domain.py` le lit comme avant. Voir `ios/README_mac.md`.
3. **Recalage sur PC** de scans indépendants (deux sessions sans carte, deux téléphones, un ancien scan) :
   ```bat
   python register_scans.py data\salon\keyframes data\chambre\keyframes [autres scans...]
   python scan_to_domain.py data\salon\keyframes data\chambre\keyframes --out data\appart --name appart
   ```
   `register_scans.py` écrit `alignment.json` et `alignment.png` (vue de dessus : référence en gris, scan recalé en
   rouge, recouvrement en noir) à côté de chaque scan recalé. La verticale est connue (gravité ARKit), le sol donne
   le décalage vertical, l'orientation des murs la rotation à un quart de tour près ; la translation vient de la
   corrélation des vues de dessus, et chaque candidat est affiné par ICP point-plan (Open3D), le meilleur recouvrement
   gagne. Il faut une zone commune (une pièce, un couloir) aux deux scans ; en deçà de 30 % de recouvrement un
   avertissement invite à vérifier `alignment.png`. `--global` ajoute un candidat sans a priori (FPFH + RANSAC) quand
   les murs ne dominent pas. `scan_to_domain.py` applique la transformation aux poses du scan recalé, sculpte un seul
   domaine et fusionne les `mesh.ply` dans `mesh_merged.ply` (`"scan.mesh"` du cas) pour les portes, fenêtres et murs.
4. **Jumeau numérique** : garder chaque conversion dans son dossier (`data\salon_2026-11`, `data\salon_2027-03`…) et
   comparer deux versions, à condition qu'elles partagent le repère ARKit (scan repris, ou second scan recalé puis
   converti avec le premier en référence) :
   ```bat
   python compare_scans.py data\salon_v1\salon.json data\salon_v2\salon.json --out data\salon_diff
   ```
   Sorties : `diff.npy` (0 air, 1 solide inchangé, 2 apparu, 3 disparu, 4 non comparé, sur la grille de la
   référence), `diff.json` (volumes, zones changées avec leur boîte englobante), `apercu_diff.png` (vert apparu,
   rouge disparu), `rapport_diff.txt`. `--tol` (défaut 1 maille) absorbe le bruit de recalage.

## Avec l'application SpatialCapture

SpatialCapture filme en vidéo LiDAR : une carte de profondeur ARKit (`sceneDepth`, 256×192, en mètres) et une pose
par image, à 30 images/s. Extraire le zip, puis donner le dossier de la prise au convertisseur, qui reconnaît le
format par `manifest.json` :

```bat
python scan_to_domain.py data\Take-2026-11-07T20-57-18Z-A5EC52 --out data\prise1 --name prise1 --dx 0.05
```

- Seules les images à suivi « normal » sont lues ; une prise sans LiDAR (`depthCaptureMode` = `none`) est refusée.
- Les images successives d'une vidéo sont presque identiques : on garde une image dès que le téléphone a bougé de
  `--move` m (défaut 0,05) ou tourné de `--turn` degrés (défaut 5). Prise de 52 s : 1501 images, 156 gardées,
  conversion en ~35 s.
- Les photos `rgb/*.jpg` (2 par seconde) donnent la couleur des surfaces : chaque pixel LiDAR coloré est projeté dans sa
  maille, la moyenne pondérée par l'angle de vue devient la couleur de la maille solide (`colors.npy`, affichée par le
  moteur dans la vue 3D, case « Couleurs du scan »). Les faces jamais photographiées prennent la couleur de la surface
  mesurée la plus proche. `video.mov`, `imu.jsonl` et `gps.jsonl` ne servent pas.
- Conseils de prise : marcher lentement, balayer les murs de haut en bas et viser le plafond ; le LiDAR porte à
  ~5 m, les vitres et les miroirs ne renvoient rien.

## Ce que fait le convertisseur

- Chaque pixel LiDAR fiable devient un point du monde (pose ARKit, échelle métrique, verticale par la gravité).
- Les murs sont alignés sur les axes x, y (histogramme des normales des surfaces verticales).
- Sol = plus bas plan horizontal tourné vers le haut ; plafond = plus haut tourné vers le bas, sinon la hauteur estimée
  par RoomScan (`scan.json`), sinon `--height` ou 2,50 m.
- Sculpture de l'espace libre : les mailles traversées par un rayon LiDAR sont de l'air, celle où il s'arrête est une
  surface. Mailles jamais vues : air (`--unknown solid` pour l'inverse). Une dalle solide ferme le sol et le plafond ;
  les poches d'air coupées de la zone parcourue sont rebouchées.

- Portes et fenêtres (si le scan contient le `mesh.ply` de RoomScan) : les triangles classés « door » et « window » par
  ARKit sont projetés sur la grille, regroupés par mur, filtrés (au moins 0,25 m² dans le plan du mur, 0,6 m de haut,
  0,4 m de large : ARKit se trompe parfois sur un meuble vitré) et régularisés en un rectangle d'une maille d'épaisseur,
  posé sur la couche du mur qui touche l'air de la pièce. `apercu_ouvertures.png` les montre vus depuis la pièce. Un
  battant ouvert, hors du plan du mur, est ignoré. Pour un domaine déjà converti :
  `python detect_openings.py <dossier>\salon.json` (le `mesh.ply` est cherché à côté de `"scan.source"`, sinon `--mesh`).

Options utiles : `--dx` (taille des mailles), `--min-conf` (confiance LiDAR 0/1/2), `--min-hits`, `--close`, `--stride`.

## Tests

```bat
python -m pytest tests -q
```

## Limites

- Les vitres ne sont pas reconnues : la fenêtre est un mur ou un trou selon ce que le LiDAR a vu (la valeur 2
  « vitre » du masque est prévue par le moteur mais n'est pas encore produite).
- Les zones non scannées sont de l'air par défaut : scanner tout le volume, plafond compris.
- L'application iOS a été écrite sans Mac : la première compilation peut demander de petites corrections.
- RoomScan n'écrit plus de `mesh.obj` : le maillage ARKit est dans `mesh.ply` (les convertisseurs ne lisent que
  `keyframes/` et `scan.json`, les anciens scans restent lisibles).
- Scan repris ou scans recalés : une zone scannée deux fois figure deux fois dans le maillage (les poses d'ARKit ne
  sont pas corrigées rétroactivement ; sa dérive est de quelques centimètres par pièce). Le maillage fin
  (`scan_to_mesh.py`) n'est construit qu'à partir du scan de référence.
