#!/bin/bash

echo "🔍 Vérification des fichiers TripoSR"
echo "====================================="

# Vérifier que les fichiers existent
echo ""
echo "📁 Fichiers requis:"

files=(
    "triposr_reconstruction.py"
    "install_triposr.sh"
    "main_gaussian.py"
    "start_gaussian.sh"
)

all_ok=true

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MANQUANT)"
        all_ok=false
    fi
done

# Vérifier les imports dans main_gaussian.py
echo ""
echo "🔍 Vérification des imports:"

if grep -q "from triposr_reconstruction import reconstruct_3d_triposr" main_gaussian.py; then
    echo "  ✅ Import TripoSR correct"
else
    echo "  ❌ Import TripoSR manquant"
    all_ok=false
fi

if grep -q "reconstruct_3d_triposr" main_gaussian.py; then
    echo "  ✅ Fonction TripoSR utilisée"
else
    echo "  ❌ Fonction TripoSR non utilisée"
    all_ok=false
fi

echo ""
if [ "$all_ok" = true ]; then
    echo "✅ Tous les fichiers sont corrects !"
    echo "🚀 Vous pouvez commit et push"
else
    echo "❌ Certains fichiers sont incorrects"
    echo "⚠️  Vérifiez les erreurs ci-dessus"
fi
