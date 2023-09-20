class UiState {
    get fen() {
        return $("#fen-input").val();
    }

    set fen(val = "") {
        $("#fen-input").val(val)
    }

    get castlingRights() {
        return [
            $("#queenside-castling-rights-black").is(":checked"),
            $("#kingside-castling-rights-black").is(":checked"),
            $("#queenside-castling-rights-white").is(":checked"),
            $("#kingside-castling-rights-white").is(":checked")
        ];
    }

    /**
     * @param {number[]} c
     */
    set castlingRights(c) {
        $("#queenside-castling-rights-black").prop("checked", c[0]);
        $("#kingside-castling-rights-black").prop("checked", c[1]);
        $("#queenside-castling-rights-white").prop("checked", c[2]);
        $("#kingside-castling-rights-white").prop("checked", c[3]);
    }

    get halfMoves() {
        let halfMoves = $("#halfmoves-counter").val();
        // Special case, just to make sure an "OK" value is given if the value in the input-field
        // is removed (i.e. an empty string)
        if (halfMoves == "") {
            halfMoves = 0;
        }
        return parseInt(halfMoves);
    }

    /**
     * @param {number} halfMoves
     */
    set halfMoves(halfMoves) {
        $("#halfmoves-counter").val(halfMoves);
    }
}