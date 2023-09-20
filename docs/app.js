class App {
    constructor(model) {
        this.model = model;
        // Saving the state in the UI :)
        this.playerToMove = 0;
        this.uiState = new UiState();
        var config = {
            draggable: true,
            position: "start",
            dropOffBoard: "trash",
            sparePieces: true,
            onChange: (oldPos, newPos) => {
                this.predict(newPos);
            }
        };
        this.board = Chessboard("board1", config);

        addOverlays();
        this.predict();

        // Resizing the board also removes the overlays.
        // Want to re-add the overlays, but don't want to do that for each
        // (incremental) resize-operation. 
        let resizeTimeout;
        $(window).resize(() => {
            this.board.resize();
            if (resizeTimeout) {
                clearTimeout(resizeTimeout);
            }
            resizeTimeout = setTimeout(() => {
                addOverlays();
                this.predict();
            }, 100);
        });
    }

    predict(pos = null) {
        if (pos == null) {
            pos = this.board.position();
        }
        // Additional info (player to move etc.) should always be updated
        // on input. Don't have to check for it again.
        let sample = boardToSample(pos, this.playerToMove, this.uiState.castlingRights, this.uiState.halfMoves);
        let prediction = this.model.predict(tf.tensor([sample], [1, 20, 8, 8]));
        prediction = tf.reshape(prediction, [12, 8, 8]).arraySync();
        updateColors(predictedMaskToVisual(sample, prediction));
    }

    readFEN(fen) {
        let splits = fen.split(" ");
        // If the FEN also specifies new castling rights, half-turns, and player to move
        if (splits.length > 1) {
            let newPlayerToMove = splits[1] == "w" ? 0 : 1;
            if (newPlayerToMove != this.playerToMove) {
                this.flipBoard();
            }
            this.uiState.castlingRights = [
                (splits[2].indexOf("q") !== -1) ? 1 : 0,
                (splits[2].indexOf("k") !== -1) ? 1 : 0,
                (splits[2].indexOf("Q") !== -1) ? 1 : 0,
                (splits[2].indexOf("K") !== -1) ? 1 : 0,
            ];
            this.uiState.halfMoves = parseInt(splits[4]);
        }
        this.board.position(splits[0]);
    }

    resetBoard() {
        this.board.start();
        this.predict();
    }

    clearBoard() {
        this.board.clear();
        this.predict();
    }

    flipBoard() {
        this.board.flip();
        this.playerToMove = this.playerToMove == 1 ? 0 : 1;
        addOverlays();
        this.predict();
    }

    updateCastlingRights() {
        this.castlingRights = this.uiState.castlingRights;
        this.predict();
    }

    processFENInput() {
        this.readFEN(this.uiState.fen);
        this.predict();
    }

    clearFENInput() {
        this.uiState.fen = "";
    }
}

