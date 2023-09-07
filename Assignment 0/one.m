I = eye(10);
A = [zeros(2,2), rand(2,2)];
a = A(:,2);
b = reshape(ones(10,1)*ones(1,10), [1,100]);
a = sort(rand(1,100));
b = a([end:-1:1]);
[u,v,w] = svd(rand(3,3));