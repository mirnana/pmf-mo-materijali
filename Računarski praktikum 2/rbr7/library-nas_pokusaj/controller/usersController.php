<?php

require_once __DIR__ . '/../model/libraryservice.class.php';

class UsersController {
    function index()  {
        $servis = new LibraryService();
        $title = 'naslov';
        $userList = $servis->getAllUsers();
        require_once __DIR__ . '../view/users_index.php';
    }
}

?>