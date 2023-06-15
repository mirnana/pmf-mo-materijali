<?php 

require_once __DIR__ . '/../../model/libraryservice.class.php';
require_once __DIR__ . '/../../model/user.class.php';

$ls = new LibraryService();
$users = $ls->getAllUsers();

echo '<pre>';
print_r( $users );
echo '</pre>';

?>
