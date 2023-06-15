<?php require_once __DIR__ . '/_header.php'; ?> 

<table>
    <tr>
        <th>Ime</th>
        <th>Prezime</th>
        <th>Posudbe</th>
    </tr>

    <?php
        foreach( $userList as $user )
        {
            echo '<tr>';
            echo '<td>' . $user->name . '</td>';
            echo '<td>' . $user->surname . '</td>';
            echo '<td><a href="index.php?rt=loans/byUser&id=' . $user->id . '">Link na posudbe</a></td>';
            echo '</tr>';
        }
    ?>
</table>

<?php require_once __DIR__ . '/_footer.php'; ?>
