<?php
require_once __DIR__ . '/../app/database/db.class.php';

class LibraryService {
    function getAllUsers() {
        $db = DB::getConnection();
        $st = $db->prepare("SELECT * FROM users");
        $st -> execute;

        $users = [];
        while($row = $st->fetch()){
            $users[] = new User($row['id'], $row['name'], $row['surname'], $row['password']);
        }

        return $users;
        
    }
    
}

?>