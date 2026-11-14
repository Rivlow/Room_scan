# RoomScan : compiler et installer sur l'iPhone (depuis un Mac)

Application iPhone (iOS 16+, LiDAR requis : iPhone 12 Pro et suivants « Pro ») qui scanne une pièce avec le maillage
temps réel d'ARKit et exporte :
- `keyframes/` : images 1024×768, profondeur LiDAR 16 bits (mm), confiance, poses — le format lu par
  `scan_to_domain.py` ;
- `mesh.ply` : binaire petit-boutiste, repère ARKit, en mètres. Sommet 24 octets (x y z nx ny nz en float32), face
  14 octets (`uchar 3`, trois `int32`, `uchar` classe). Classes : 0 aucune, 1 mur, 2 sol, 3 plafond, 4 table, 5 siège,
  6 fenêtre, 7 porte. Plus de `mesh.obj` ni de `mesh.mtl` ;
- `missing.ply` : le voile des zones manquantes en fin de scan (même format, classe 0), montré en rouge dans l'aperçu ;
- `worldmap.arworldmap` : la carte ARKit (`ARWorldMap` archivée, dizaines de Mo pour un appartement) et `coverage.bin`
  (la grille des zones manquantes) : ce qu'il faut pour **reprendre le scan** plus tard dans le même repère ;
- `scan.json` : appareil, durée, nombre d'images clés (et `keyframesSkipped`, tentatives ignorées car la file d'écriture
  était pleine) et de triangles, hauteur sol / plafond estimée, `peakFootprintMB`, `sessions` (1, puis +1 par reprise),
  `worldMapBytes` (absent si la carte n'a pas pu être sauvegardée) ;
- `capture_log.txt` : une ligne toutes les 2 s (`t= kf= depth= max= skipFull= fail= thermal= footprint= free= tri=`), plus
  les alertes mémoire, les changements thermiques et l'empreinte avant / après l'export.

Comportements automatiques pendant le scan (affichés dans le bandeau, le maillage continue dans tous les cas) :
- téléphone chaud : cadence des images clés divisée par 2 (`serious`) ou par 4 avec maillage masqué (`critical`) ;
- alerte mémoire : pas d'image clé pendant 5 s ; deux alertes en 30 s arrêtent la capture des images ;
- moins de 0,5 Go libres : la capture des images s'arrête ;
- file d'écriture pleine (4 images en attente) : l'image clé est ignorée plutôt que de gonfler la mémoire.

## 1. Préparer le Mac (une fois)

1. Installer **Xcode** depuis l'App Store (15 ou plus récent), l'ouvrir une fois pour accepter la licence et laisser
   installer les composants iOS.
2. Installer XcodeGen (génère le projet Xcode à partir de `project.yml`) :
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"   # si Homebrew manque
   brew install xcodegen
   ```
3. Copier le dossier `Room_scan/ios/` sur le Mac (AirDrop, clé USB, git…).

## 2. Générer et ouvrir le projet

```bash
cd ios
xcodegen            # crée RoomScan.xcodeproj
open RoomScan.xcodeproj
```

Sans XcodeGen : Xcode > File > New > Project > iOS App (nom RoomScan, interface SwiftUI, langage Swift), supprimer
les fichiers Swift générés, glisser le dossier `RoomScan/Sources` dans le projet (cocher « Copy items if needed »),
puis dans l'onglet Info de la cible ajouter les clés de `RoomScan/Info.plist` (au minimum
`NSCameraUsageDescription`, `UIFileSharingEnabled`, `LSSupportsOpeningDocumentsInPlace`, `UIRequiredDeviceCapabilities`
= `arkit`).

## 3. Signer avec un identifiant Apple gratuit

1. Xcode > Settings > Accounts > « + » > ajouter l'identifiant Apple (le compte iCloud suffit).
2. Dans le projet : cible **RoomScan** > onglet **Signing & Capabilities** > cocher « Automatically manage signing »,
   choisir l'équipe « (Personal Team) ».
3. Si Xcode se plaint de l'identifiant de bundle, changer `fr.lucas.roomscan` en quelque chose d'unique
   (par exemple `fr.votreprenom.roomscan`) dans **General > Identity**.

## 4. Installer sur l'iPhone

1. Brancher l'iPhone en USB, le déverrouiller, accepter « Se fier à cet ordinateur ».
2. Sur l'iPhone : Réglages > Confidentialité et sécurité > **Mode développeur** > activer (redémarrage).
3. Dans Xcode, choisir l'iPhone comme destination (barre du haut) puis **Run** (⌘R).
4. Premier lancement : l'iPhone refuse l'application « développeur non approuvé » : Réglages > Général >
   **VPN et gestion de l'appareil** > votre identifiant > Se fier.

Limites d'un compte gratuit : l'application expire **au bout de 7 jours** (il suffit de refaire ⌘R depuis Xcode),
trois applications au plus. Avec un compte développeur payant (99 €/an) : un an, et TestFlight sans câble.

## 5. Scanner

- Lumière allumée, portes ouvertes, objectif propre.
- Démarrer, attendre « Suivi correct », puis marcher **lentement** le long des murs à 1 à 3 m des surfaces (portée du
  LiDAR ~5 m), en balayant du sol au plafond. Le maillage coloré se dessine en temps réel : ce qui n'est pas dessiné
  n'est pas scanné. Viser explicitement le plafond et le sol de chaque pièce (l'app le rappelle).
- Un passage à hauteur de poitrine puis un second plus bas (dessous des tables, derrière les meubles).
- **Visualisation** (bouton en haut à droite, changeable à tout moment, sans effet sur ce qui est enregistré) :
  - *Caméra + maillage* (défaut) : le maillage ARKit coloré par classe sur l'image ; ce qui n'est pas dessiné n'est
    pas scanné ;
  - *Caméra* : l'image seule (moins de chauffe) ;
  - *Nuage de points* : fond noir, un point LiDAR par maille de 2,5 cm, coloré par hauteur (bleu bas → jaune haut),
    les points de la dernière seconde brillent en blanc : on voit la pièce se construire et les trous de couverture ;
  - *Confiance LiDAR* : fond noir, points verts (confiance haute), jaunes (moyenne), rouges (faible) : où les
    mesures sont douteuses (vitres, surfaces noires, bords, trop loin) ;
  - *Caméra + nuage* : les points par hauteur sur l'image, pour vérifier que le nuage colle à la réalité.
  Le nuage (jusqu'à 1,5 million de points, ~24 Mo, un pixel de profondeur sur 4 toutes les 0,2 s) est dessiné par un
  shader Metal en post-traitement de la vue RealityKit, compilé au lancement ; s'il est refusé, le bandeau le dit et les
  modes caméra restent disponibles. Le voile rouge des zones manquantes reste visible dans tous les modes.
- **Voile rouge des zones manquantes** (bouton triangle rouge, activé par défaut) : là où le regard s'est arrêté sans
  rien rencontrer (mur pas encore scanné, coin oublié, trou). Grille de 10 cm nourrie par la profondeur LiDAR pendant
  l'enregistrement ; il disparaît dès que la zone est visée. Pas de voile près d'une porte ou d'une fenêtre reconnue par
  ARKit, ni pour les zones de moins de 0,1 m² (souvent sous ou derrière un meuble). Le bandeau donne la surface totale.
  Limites : une vitre vers l'extérieur peut en montrer au-delà ; une maille inconnue collée à une surface vue est comptée
  comme « derrière le mur », ce qui rétrécit chaque trou d'une maille sur son pourtour.
- **Pause** : le suivi ARKit continue (le repère est conservé) ; on peut marcher jusqu'à une autre pièce et reprendre
  exactement là. L'écran reste allumé, le maillage en surimpression est masqué pour moins chauffer, et **Voir le scan**
  montre le maillage complet et le voile en 3D (orbite au doigt). Une fermeture de l'application ou une batterie vide
  termine en revanche la session : le scan ne peut pas être repris ensuite.
- Éviter de s'arrêter longtemps devant un miroir ou une grande vitre.
- Terminer : l'export prend quelques secondes (jusqu'à une minute pour un appartement complet). La carte ARKit est
  demandée juste avant l'arrêt de la session : le bandeau « Carte ARKit : limitée / en extension / complète » dit
  si elle vaut quelque chose (terminer sur un suivi correct, après avoir balayé lentement).
- **Reprendre le scan** (fiche du scan > Reprendre le scan) : à un autre moment, batterie rechargée, après avoir
  fermé l'app. La carte est rechargée (quelques secondes), le suivi démarre en « Relocalisation » : revenir à un endroit
  déjà scanné, dans un éclairage comparable, et balayer lentement jusqu'à « Suivi correct ». Rien n'est enregistré
  avant. Ensuite tout s'ajoute au même scan et au même repère : images clés, maillage (fusionné à l'ancien `mesh.ply`
  à l'export, une zone rescannée y figure deux fois), zones manquantes (grille rechargée : le voile montre encore ce
  qui manquait), `scan.json` (compteurs cumulés, `sessions`). Annuler pendant une reprise efface seulement les images
  clés de cette session. Limites : ARKit ne retrouve pas le repère si la pièce a beaucoup changé, si l'éclairage est
  très différent ou si l'on démarre dans une pièce jamais scannée ; on peut alors scanner séparément et recaler sur
  PC (`register_scans.py`, voir le README principal).

Volume : environ 4 images clés par seconde de déplacement, ~200 Ko chacune ; un appartement de 10 minutes ≈ 500 Mo.

## 6. Récupérer le scan sur le PC

- Dans l'app : scan > **Partager** > AirDrop vers le Mac (archive zip, créée dans le dossier temporaire et supprimée
  à la fermeture de la feuille de partage ; il faut autant de place libre que la taille du scan), ou
- Fichiers (iPhone) > Sur mon iPhone > RoomScan > Scans > dossier du scan, ou
- câble : Finder / iTunes > iPhone > Fichiers > RoomScan.

Puis sur le PC :
```bat
python scan_to_domain.py <dossier du scan>\keyframes --out <dossier de sortie> --name salon
```
(`--height 2.50` si le plafond n'a pas été scanné ; le script lit sinon la hauteur estimée dans `scan.json`.)

## 7. Tests (sans iPhone)

Le calcul des zones manquantes (`CoverageMap.swift`, Swift pur) est testé sous Linux sur une pièce synthétique :
`wsl -d Ubuntu-24.04 -- bash Room_scan/ios/tests/run_tests.sh`.

## 7 bis. Sans Mac : compiler et installer depuis WSL (xtool)

Mise en place une fois : `setup_wsl.sh system` puis `setup_wsl.sh user`, `xtool setup` (identifiant Apple, Xcode.xip), et
côté Windows iTunes (service Apple Mobile Device) avec un portproxy `<passerelle WSL>:27015 -> 127.0.0.1:27015` et une
règle de pare-feu entrante sur 27015. Ensuite, iPhone branché et déverrouillé :

```powershell
wsl -d Ubuntu-24.04 -- bash C:\...\Room_scan\ios\build_wsl.sh run      # build : compile seulement
```

Si xtool reste sur « Waiting for device to be connected… » alors que Windows voit l'iPhone : le portproxy n'est pas
monté (fréquent après un redémarrage de Windows, l'adaptateur WSL apparaît après le service IP Helper). Le script le
détecte et s'arrête ; remède, dans un PowerShell administrateur : `Restart-Service iphlpsvc -Force`. Si l'adresse de la
passerelle WSL a changé (`ip route list default` dans WSL), refaire le portproxy avec la nouvelle adresse.

## 8. En cas d'erreur de compilation

Le code a été écrit sans Mac sous la main : si Xcode signale une erreur, copier le message complet (fichier, ligne,
texte) et le renvoyer ; la correction est en général d'une ligne.
