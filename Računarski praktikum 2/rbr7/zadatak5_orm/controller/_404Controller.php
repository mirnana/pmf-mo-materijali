<?php 

require_once __DIR__ . '/../model/libraryservice.class.php';

class _404Controller
{
    public function index()
    {
        $title = 'Pristupili ste nepostojećoj stranici.';

        require_once __DIR__ . '/../view/_404_index.php';
    }
}

?>
