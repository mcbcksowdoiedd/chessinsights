from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, HttpResponseBadRequest
from .forms import UploadFileForm
from .models import UploadedFile
import chess.pgn
import chess.engine
from sklearn.cluster import KMeans
import numpy as np
import mimetypes
import os

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            player_color = request.POST.get('player_color', 'white')  # Default to white if not specified
            uploaded_file = form.save(commit=False)
            uploaded_file.player_color = player_color
            uploaded_file.save()
            print("File Saved Successfully:", uploaded_file.file.name)
            return redirect('upload_success')
        else:
            print("Form is not valid:", form.errors)
            return HttpResponseBadRequest("Form submission failed. Please check the form errors.")
    else:
        form = UploadFileForm()
    return render(request, 'filehandler/upload.html', {'form': form})

def analyze_pgn_and_get_results(pgn_file, player_to_analyze):
    engine = chess.engine.SimpleEngine.popen_uci("/home/akanksha/Downloads/stockfish-ubuntu-x86-64/stockfish/stockfish-ubuntu-x86-64")

    centipawn_losses = []
    total_moves = 0
    with open(pgn_file) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            player_moves, moves_count = calculate_centipawn_loss(game, engine, player_to_analyze)
            centipawn_losses.extend(player_moves)
            total_moves += moves_count

    engine.quit()

    centroids = perform_clustering(centipawn_losses)
    average_centroid = np.mean(centroids)

    beginner_threshold = 0.20
    intermediate_threshold = 0.40
    professional_threshold = 0.70

    category = categorize_players(average_centroid, beginner_threshold, intermediate_threshold, professional_threshold)

    learning_resources = {
        "Beginner": ["Chess Basics Tutorial", "Pawn Structure Strategies", "Opening Principles"],
        "Intermediate": ["Tactics Training", "Middle Game Planning", "Endgame Essentials"],
        "Expert": ["Advanced Tactics and Combinations", "Strategic Planning", "Endgame Mastery"],
        "Professional": ["Grandmaster Game Analysis", "Advanced Opening Theory", "Positional Sacrifices"]
    }

    return {
        "total_moves": total_moves,
        "average_centroid": average_centroid,
        "category": category,
        "learning_resources": learning_resources.get(category, [])
    }

def calculate_centipawn_loss(game, engine, player_color):
    board = game.board()
    centipawn_losses = []
    total_moves = 0

    for move in game.mainline_moves():
        if (player_color == "white" and board.turn == chess.WHITE) or \
           (player_color == "black" and board.turn == chess.BLACK):
            total_moves += 1
            info = engine.analyse(board, chess.engine.Limit(time=1.0))

            if info["score"].relative is not None and not isinstance(info["score"].relative, chess.engine.Mate):
                centipawn_loss = abs(info["score"].relative.cp)
                centipawn_losses.append(centipawn_loss)

        board.push(move)

    return centipawn_losses, total_moves

def perform_clustering(centipawn_losses):
    X = np.array(centipawn_losses).reshape(-1, 1)
    X_normalized = (X - X.mean()) / X.std()
    n_clusters = min(len(centipawn_losses), 2)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_normalized)
    centroids = kmeans.cluster_centers_
    return centroids

def categorize_players(average_centroid, beginner_threshold, intermediate_threshold, professional_threshold):
    if average_centroid < beginner_threshold:
        return "Beginner"
    elif average_centroid < intermediate_threshold:
        return "Intermediate"
    elif average_centroid < professional_threshold:
        return "Professional"
    else:
        return "Expert"

def download_file(request):
    latest_file = UploadedFile.objects.last()
    analysis_results = None
    
    if latest_file:
        pgn_file_path = latest_file.file.path
        analysis_results = analyze_pgn_and_get_results(pgn_file_path, player_to_analyze=latest_file.player_color)
        print("Analysis Results:", analysis_results)

    return render(request, 'filehandler/download.html', {'latest_file': latest_file, 'analysis_results': analysis_results})

def upload_success(request):
    return render(request, 'filehandler/upload_success.html')

def download_pdf(request, file_id):
    uploaded_file = get_object_or_404(UploadedFile, id=file_id)
    category = request.GET.get('category')
    pdf_file_path = None
    
    if category:
        pdf_folder = os.path.join(os.path.dirname(__file__), 'resources')
        if category == 'Beginner':
            pdf_file_path = os.path.join(pdf_folder, 'Beginners Guide.pdf')
        elif category == 'Intermediate':
            pdf_file_path = os.path.join(pdf_folder, 'INTERMEDIATE Guide.pdf')
        elif category == 'Expert':
            pdf_file_path = os.path.join(pdf_folder, 'EXPERT Guide.pdf')
        elif category == 'Professional':
            pdf_file_path = os.path.join(pdf_folder, 'Professional Guide.pdf')

    if pdf_file_path and os.path.exists(pdf_file_path):
        with open(pdf_file_path, 'rb') as pdf_file:
            response = HttpResponse(pdf_file.read(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf_file_path)}"'
            return response
    else:
        return HttpResponseBadRequest("PDF file not found for the player's category.")
