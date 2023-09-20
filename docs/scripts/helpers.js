const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
const ranks = ["1", "2", "3", "4", "5", "6", "7", "8"];


function updateColors(prediction) {
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            let square = files[i] + ranks[j];
            let color = prediction[j][i] > 0.5 ? "#58C4DD" : "#FC6255";
            let opacity = Math.abs(0.5 - prediction[j][i]) * 2;
            let element = $("#square-" + square + "-overlay");
            element.css("background-color", color);
            element.css("opacity", opacity);
        }
    }
}


function addOverlays() {
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            let square = files[i] + ranks[j];
            let element = $(".square-" + square);
            element.append("<div id='square-" + square + "-overlay' class='overlay'></div>");
        }
    }
}

function boardToSample(board, playerToMove, castlingRights, halfMoves) {
    let sample = emptySample(0);

    let pieceToPlaneIndex = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5
    };

    // sample[12] = queenside castling for black
    // sample[13] = kingside castling for black
    // sample[14] = queenside castling for white
    // sample[15] = kingside castling for white
    // sample[16] = player to move has black pieces
    // sample[17] = halfmoves since last capture
    // sample[18] = zeroes
    // sample[19] = ones

    // The player to move's castling rights should be first
    if (playerToMove) {
        sample[12] = plane(castlingRights[0] ? 1 : 0);
        sample[13] = plane(castlingRights[1] ? 1 : 0);
        sample[14] = plane(castlingRights[2] ? 1 : 0);
        sample[15] = plane(castlingRights[3] ? 1 : 0);
    } else {
        sample[12] = plane(castlingRights[2] ? 1 : 0);
        sample[13] = plane(castlingRights[3] ? 1 : 0);
        sample[14] = plane(castlingRights[0] ? 1 : 0);
        sample[15] = plane(castlingRights[1] ? 1 : 0);
    }
    sample[16] = plane(playerToMove);
    sample[17] = plane(halfMoves);
    sample[18] = plane(0);
    sample[19] = plane(1);

    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            let square = files[i] + ranks[j];
            if (board.hasOwnProperty(square)) {
                let pieceIsBlack = board[square][0] == "b";
                let offset = (pieceIsBlack === !playerToMove) ? 6 : 0;
                let planeIndex = offset + pieceToPlaneIndex[board[square][1]];
                if (playerToMove) {
                    sample[planeIndex][7 - j][i] = 1;
                } else {
                    sample[planeIndex][j][i] = 1;
                }
            }
        }
    }
    return sample;
}

function predictedMaskToVisual(board, prediction) {
    // The model now predicts a tensor with the shape (12, 8, 8), i.e. a 12d vector for each
    // square. Each element in this vector corresponds to a given type of piece (piece-type + color).
    // The mask is created as follows:
    // For the squares that have no piece, the overlay shows the value corresponding to the largest element in the 12d-vector
    // for the given square. I.e. the highest value for any type of piece for the given square.
    // For the squares that have a piece, the overlay shows the value of the piece that is there. I.e. the importance
    // of the piece that is on the square already.
    let overlay = [];
    for (let i = 0; i < 8; i++) {
        overlay.push([]);
        for (let j = 0; j < 8; j++) {
            let valueToPush = -1;
            for (let p = 0; p < 12; p++) {
                if (board[p][i][j]) {
                    // Has a piece at (i, j), should use its predicted importance as overlay value
                    valueToPush = prediction[p][i][j];
                }
            }
            // There was no piece at (i, j), use the max importance of any piece at (i, j) as overlay value
            if (valueToPush == -1) {
                valueToPush = getMaxFromChannel(prediction, i, j);
            }
            overlay[i].push(valueToPush);
        }
    }
    return overlay;
}