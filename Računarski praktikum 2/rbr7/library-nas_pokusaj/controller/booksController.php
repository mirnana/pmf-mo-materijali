<?php

require_once __DIR__ . '/../model/libraryservice.class.php';

class BooksController {
    function index()  {
        $servis = new LibraryService();
        $title = 'Popis svih knjiga';
        $bookList = $servis->getAllBooks();
        require_once __DIR__ . '../view/books_index.php';
    }

    function getAllBooks() {
        $db = DB::getConnection();
        $st = $db->prepare()
    }
}
?>