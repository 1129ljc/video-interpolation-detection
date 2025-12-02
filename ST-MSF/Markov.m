function [F] = Markov(X,T)
X = double(X);
[F] = ExtractMarkovPart_new(X,T);

function [MarkovPart] = ExtractMarkovPart_new(absDctPlane,T)
Fh=absDctPlane(:,1:end-1) - absDctPlane(:,2:end); % 舍去最后一行
Fv=absDctPlane(1:end-1,:) - absDctPlane(2:end,:); % 舍去最后一排
Fd=absDctPlane(1:end-1,1:end-1) - absDctPlane(2:end,2:end); %舍去最后一行和最后一排
Fm=absDctPlane(2:end,1:end-1) - absDctPlane(1:end-1,2:end); %
OrMh=calculateMarkovField2(Fh(:,1:end-1),Fh(:,2:end),T);
OrMv=calculateMarkovField2(Fv(1:end-1,:),Fv(2:end,:),T);
OrMd=calculateMarkovField2(Fd(1:end-1,1:end-1),Fd(2:end,2:end),T);
OrMm=calculateMarkovField2(Fm(2:end,1:end-1),Fm(1:end-1,2:end),T);

Fh1 = -Fh;
Fv1 = -Fv;
Fd1 = -Fd;
Fm1 = -Fm;
OrMh1=calculateMarkovField2(Fh1(:,1:end-1),Fh1(:,2:end),T);
OrMv1=calculateMarkovField2(Fv1(1:end-1,:),Fv1(2:end,:),T);
OrMd1=calculateMarkovField2(Fd1(1:end-1,1:end-1),Fd1(2:end,2:end),T);
OrMm1=calculateMarkovField2(Fm1(2:end,1:end-1),Fm1(1:end-1,2:end),T);

F1 = (OrMh+OrMv+OrMh1+OrMv1)/2;
F2 = (OrMd+OrMm+OrMd1+OrMm1)/2;
F11 = reshape(F1,1,[]);
F2 = reshape(F2,1,[]);
MarkovPart = [F11 F2];

function field=calculateMarkovField2(reference,shifted,T)
field=zeros(2*T+1,2*T+1);
R=reference(:);
S=shifted(:);
R(R>T)=T; R(R<-T)=-T;
S(S>T)=T; S(S<-T)=-T;
for i=-T:T
    S2 = S(R==i);
    h=hist(S2,-T:T);
    h=h/max(length(S2),1);
    field(i+T+1,:)=h;
end

