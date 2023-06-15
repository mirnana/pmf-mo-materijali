<?php require_once __DIR__.'../../view/_header.php'; ?>

<table>
    <tr>
        <th>Ime</th>
        <th>Prezime</th>
    </tr>

    <?php
        foreach($userList as $User) {
            echo '<tr>';
            echo '<td>' . $user->name . '</td>';
            echo '</tr>';
        }
    ?>
</table>

<?php require_once __DIR__.'./_footer.php'; ?>